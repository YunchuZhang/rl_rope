import torch
from torch import nn
import numpy as np

from tycho_env.utils import get_transformation_matrix_from_quat

def dh_transformation(alpha, a, _theta, d, theta_offset, device = "cpu"):
    """Given the DH link construct the transformation matrix"""
    theta = torch.add(_theta, theta_offset)
    batch_size = theta.shape[0]

    m = torch.eye(4, device=device, dtype=torch.float32).tile((batch_size, 1, 1))

    # rotation matrix
    m[:, 0, 0] = torch.cos(theta)
    m[:, 0, 1] = -torch.sin(theta)
    m[:, 1, 0] = torch.sin(theta) * np.cos(alpha)
    m[:, 1, 1] = torch.cos(theta) * np.cos(alpha)
    m[:, 1, 2] = -np.sin(alpha)
    m[:, 2, 0] = torch.sin(theta) * np.sin(alpha)
    m[:, 2, 1] = torch.cos(theta) * np.sin(alpha)
    m[:, 2, 2] = np.cos(alpha)

    # translation vector
    m[:, 0, -1] = a
    m[:, 1, -1] = d * -np.sin(alpha)
    m[:, 2, -1] = d * np.cos(alpha)

    return m


def fk_transformation(fk_params, joint_positions, device = "cpu"):
    """Given a list of FK params (B by 4), return the transformation"""
    batch_size = joint_positions.shape[0]
    joint_positions = joint_positions.transpose(0, 1)  # num_joint x batch_size
    ee = torch.eye(4, device=device, dtype=torch.float32).tile((batch_size, 1, 1))
    for (alpha, a, d, offset), theta in zip(fk_params, joint_positions):
        # apply the transformation for each joint
        ee = torch.matmul(ee, dh_transformation(alpha, a, theta, d, offset, device))
    return ee


def init_weights(m):
    if type(m) == nn.Linear:
        # xavier initialization, set bias to 0
        nn.init.xavier_uniform_(m.weight, nn.init.calculate_gain("sigmoid"))
        nn.init.constant_(m.bias, 0)


class ResEstimator:
    """
    max_backlash: maximum backlash for all joints
    fk_params: 6 x 4 matrix. The i-th row gives the DH params for the i-th joint
    last_joint: transformation from the 6-th joint to the tip of chopsticks
    """
    def __init__(self, device, max_backlash, fk_params, last_joint,
                loss_fn=None, optimizer=None):
        self.device = device
        self.max_backlash = max_backlash
        self.fk_params = fk_params
        self.last_joint = last_joint
        self.loss_fn = loss_fn or nn.MSELoss()
        self.last_joint_trf = torch.tensor(
            get_transformation_matrix_from_quat(last_joint), device=device).float()
        self.layers = nn.Sequential(
        nn.Linear(6, 32),
        nn.Sigmoid(),
        nn.Linear(32, 32),
        nn.Sigmoid(),
        nn.Linear(32, 6),
        nn.Sigmoid()
        ).to(device)
        self.layers.apply(init_weights)

    @staticmethod
    def static_load(path, device):
        model = ResEstimator(device, 1, np.ones((6,4)), np.ones((7,)))
        model.load(path)
        return model

    def predict(self, joint_positions):
        """ Given the joint positions predict the backlash """
        jp_tensor = torch.tensor(joint_positions, device=self.device, dtype=torch.float32).unsqueeze(0)
        return self.compute_backlash(jp_tensor).detach().cpu().numpy().squeeze(0)

    def compute_backlash(self, joint_positions):
        """ Given a batch of joint positions predict the backlash """
        return torch.mul(2 * self.layers(joint_positions) - 1, self.max_backlash)
    
    def corrected_pos(self, joint_positions):
        backlash = self.compute_backlash(joint_positions)
        corrected_jp = joint_positions + backlash
        fk_arm = fk_transformation(self.fk_params, corrected_jp, self.device)
        tip_pos = (fk_arm @ self.last_joint_trf)[:, 0:3, 3]
        return tip_pos

    def compute_loss(self, joint_positions, labels):
        tip_pos = self.corrected_pos(joint_positions)
        return self.loss_fn(tip_pos, labels)

    def train(self):
        self.layers.train()

    def eval(self):
        self.layers.eval()

    def parameters(self):
        return self.layers.parameters()

    def save(self, path, optimizer, backwards_compatible=False):
        state = {}
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["model_state_dict"] = self.layers.state_dict()
        state["model_max_backlash"] = self.max_backlash
        state["model_fk_params"] = self.fk_params
        state["model_last_joint"] = self.last_joint
        torch.save(state, path, _use_new_zipfile_serialization=(not backwards_compatible))

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.layers.load_state_dict(state["model_state_dict"])
        self.max_backlash = state["model_max_backlash"]
        self.last_joint = state["model_last_joint"]
        self.last_joint_trf = torch.tensor(
            get_transformation_matrix_from_quat(self.last_joint),
            device=self.device).float()
        self.fk_params = state["model_fk_params"]
        return state
