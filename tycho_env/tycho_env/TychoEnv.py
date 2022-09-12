'''
Largely inspired by OpenAI gym mujoco_env.py at commit a14e1c7
https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
'''

from os import stat
import os.path
from functools import partial
import numpy as np
import mujoco_py
from mujoco_py import load_model_from_path, MjSim
from scipy.spatial.transform import Rotation as scipyR

import gym
from gym import spaces # consider remove?
from gym.utils import seeding # consider remove?

from .TychoController import TychoController
from .utils import construct_choppose, construct_command, \
    get_transformation_matrix_from_quat, euler_angles_from_rotation_matrix, euler_angles_to_quat
from .utils import print_and_cr

# MOVING_POSITION = [-2.6660487490, 1.4716092544, 1.8969803405, 0.4229468198, 0.4416592515, 0.0023800291, -0.3762229634]
# MOVING_POSITION = [-2.3185521204, 1.5607393456, 2.1280585854, 0.5546104157, 0.8188435375, -0.0000604033, -0.3613537596]
MOVING_POSITION = [-2.2683857389667805, 1.5267610481188283, 2.115358222505881, 0.5894993521468314, 0.8740650991816328, 0.0014332898815118368, -0.36]
DEFAULT_SIZE = 100

# DEFAULT_BALL_RAD = 0.0055
DEFAULT_BALL_RAD = 0.01
class Task:
    REACH = "reach"
    GRASP = "grasp"
    LIFT = "lift"

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def randomize_pos_above_workspace(center, dims):
    return center + (np.random.rand(len(dims)) - 0.5) * (np.array(dims) / 2)

class TychoEnv(gym.Env):
    # ============================================
    # Initialization
    def __init__(self,
        model_xml=os.path.join(os.path.dirname(__file__), "assets", "hebi_rope2.xml"),
        gains_xml=os.path.join(os.path.dirname(__file__), "assets", "chopstick-gains-7D-20220805.xml"),
        # gains_xml=os.path.join(os.path.dirname(__file__), "assets", "chopstick-gains-7D-hardwarePID.xml"),
        frame_skip=1,  timestep=0.002, control_repeat=25, # aiming for 500Hz internal control and 100Hz step frequency
        # change step freq to be 40hz? 500 / 10 = 50hz
        onlyPosition=False,
        model_params_fn=os.path.join(os.path.dirname(__file__), "assets", "model_params.txt"),
        config={},
        seed=None,
        max_episode_steps=100,
        ):
        # load xml model
        if not os.path.exists(model_xml):
            print('Model xml:', model_xml)
            raise IOError("Model XML File does not exist")
        self.model = load_model_from_path(model_xml)
        self.dr =config['dr']
        if model_params_fn:
            self.model_params_fn = model_params_fn
            self.load_model_params(self.model_params_fn)
        self.model.opt.timestep = timestep
        #self.model.data.skip # TODO check the mujoco model skip? I don't think they have one?
        # construct sim and viewer
        self.sim = MjSim(self.model)
        self.frame_skip = frame_skip
        self.control_repeat = control_repeat
        # print(f"current step freq = {1 / timestep // control_repeat}")
        self.data = self.sim.data
        self.config = {
            'state_space': 'eepose-obj',
            'action_space': 'eepose',
            'action_low': [-0.40,-0.30,0.04,0,-1,0,0,-0.57],
            'action_high':[-0.37,-0.27,0.11,0,-1,0,0,-0.2],
            'task': Task.LIFT, # one value of Task
            'ball_rad': DEFAULT_BALL_RAD,
            'static_ball': True,
            # takes env and  pos, returns 8D eepose
            'reset_eepose_fn': lambda _, __: construct_choppose(self.ctrl.arm, MOVING_POSITION, target_at_ee=True)
        }
        self.config.update(config)

        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        # set up controller
        self.ctrl = TychoController(gains_xml, onlyPosition)
        self.goal = self.sim.data.site_xpos[-1] # TODO
        self.goal = [-0.38, -0.26, 0.1]
        self._max_episode_steps = max_episode_steps
        self._step_counter = 0
        self.seed(seed)
        self._reset_jointpos()
        self._set_action_space()
        self._set_observation_space()

    def load_model_params(self, fn):
        # What is in the model_params.txt depends on the julia sysid code.
        with open(fn) as f:
            try:
                content = f.read()
                params = [float(s) for s in content.splitlines() if len(s) > 0 and s[0] != '#']
            except Exception as e:
                print(e)
                params = [float(s) for s in content.split(',')]
            #print("Read params: ", params)
        # ------------------------------------------------------------------
        from collections import namedtuple
        scaling = [16., 0.001, 0.2, 23.33, 4.4843, 40., 0.001, 0.2, 44.7632, 2.3375,   3., 0.001, 0.2, 2.6611, 14.12]
        com = [-0.0101505, -0.0489759, 0.0451999, 0.24476, 0.00166405, -0.00402425, 0.227415, 0.00136712, 0.00396226, -0.0107783, -0.0503558, 0.0331911, -0.0107783, -0.0501576, 0.0329123, -0.010267, 0.0507442, 0.0379799]
        mass = [0.715, 0.882933, 0.716861, 0.415, 0.415, 0.4215]
        ModelParams = namedtuple('ModelParams',
            ['x8_damping', 'x8_armature', 'x8_frictionloss', 'x8_maxT', 'x8_ramp',
             'x8_16_damping', 'x8_16_armature', 'x8_16_frictionloss', 'x8_16_maxT', 'x8_16_ramp',
             'x5_damping', 'x5_armature', 'x5_frictionloss', 'x5_maxT', 'x5_ramp']
            )
        # ------------------------------------------------------------------
        # Note that julia has column-first, might need to transpose ...
        # import pdb;pdb.set_trace()
        mp = ModelParams(*params[:15])
        m = self.model

        if self.dr:
            m.dof_damping[[0,2]] = np.random.uniform(mp.x8_damping * scaling[0]-0.5,mp.x8_damping * scaling[0]+0.5)
            m.dof_armature[[0,2]] = np.random.uniform(mp.x8_armature  * scaling[1]-0.0005,mp.x8_armature  * scaling[1]+0.0005)
            m.dof_frictionloss[[0,2]] = np.random.uniform(mp.x8_frictionloss * scaling[2]-0.05,mp.x8_frictionloss * scaling[2]+0.05)
            m.actuator_biasprm[[0, 2], 0] = np.random.uniform(mp.x8_maxT * scaling[3]-0.1,mp.x8_maxT * scaling[3]+0.1)
            m.actuator_biasprm[[0, 2], 1] = np.random.uniform(mp.x8_ramp * scaling[4]-0.05,mp.x8_ramp * scaling[4]+0.05)

            m.dof_damping[1] = np.random.uniform(mp.x8_16_damping * scaling[5]-0.5,mp.x8_16_damping * scaling[5]+0.5)
            m.dof_armature[1] = np.random.uniform(mp.x8_16_armature * scaling[6]-0.0005,mp.x8_16_armature * scaling[6]+0.0005)
            m.dof_frictionloss[1] = np.random.uniform(mp.x8_16_frictionloss * scaling[7]-0.01,mp.x8_16_frictionloss * scaling[7]+0.01)
            m.actuator_biasprm[1, 0] = np.random.uniform(mp.x8_16_maxT * scaling[8]-0.5,mp.x8_16_maxT * scaling[8]+0.5)
            m.actuator_biasprm[1, 1] = np.random.uniform(mp.x8_16_ramp * scaling[9]-0.1,mp.x8_16_ramp * scaling[9]+0.1)

            m.dof_damping[3:7] = np.random.uniform(mp.x5_damping * scaling[10]-0.1,mp.x5_damping * scaling[10]+0.1)
            m.dof_armature[3:7] = np.random.uniform(mp.x5_armature * scaling[11]-0.0005,mp.x5_armature * scaling[11]+0.0005)
            m.dof_frictionloss[3:7] = np.random.uniform(mp.x5_frictionloss * scaling[12]-0.01,mp.x5_frictionloss * scaling[12]+0.01)
            m.actuator_biasprm[3:7, 0] = np.random.uniform(mp.x5_maxT * scaling[13]-0.05,mp.x5_maxT * scaling[13]+0.05)
            m.actuator_biasprm[3:7, 1] = np.random.uniform(mp.x5_ramp * scaling[14]-0.5,mp.x5_ramp * scaling[14]+0.5)

            params = np.array(params)
            com = np.array(com)

            # CoM
            m.body_ipos[2, 0:3] = np.random.uniform(params[15:18] + com[0:3] - np.array([0.002,0.002,0.002]), params[15:18] + com[0:3] + np.array([0.002,0.002,0.002]))
            m.body_ipos[3, 0:3] = np.random.uniform(params[18:21] + com[3:6] - np.array([0.02,0,0]), params[18:21] + com[3:6] + np.array([0.02,0,0]))
            m.body_ipos[4, 0:3] = np.random.uniform(params[21:24] + com[6:9] - np.array([0.02,0,0]), params[21:24] + com[6:9] + np.array([0.02,0,0]))
            m.body_ipos[5, 0:3] = np.random.uniform(params[24:27] + com[9:12] - np.array([0.002,0.002,0.002]), params[24:27] + com[9:12] + np.array([0.002,0.002,0.002]))
            m.body_ipos[6, 0:3] = np.random.uniform(params[27:30] + com[12:15] - np.array([0.002,0.002,0.002]),params[27:30] + com[12:15] + np.array([0.002,0.002,0.002]))
            m.body_ipos[7, 0:3] = np.random.uniform(params[30:33] + com[15:18] - np.array([0.002,0.002,0.002]),params[30:33] + com[15:18] + np.array([0.002,0.002,0.002]))

            # Mass
            m.body_mass[2] = np.random.uniform(params[33] + mass[0] - 0.05, params[33] + mass[0] + 0.05)
            m.body_mass[3] = np.random.uniform(params[34] + mass[1] - 0.05, params[34] + mass[1] + 0.05)
            m.body_mass[4] = np.random.uniform(params[35] + mass[2] - 0.05, params[35] + mass[2] + 0.05)
            m.body_mass[5] = np.random.uniform(params[36] + mass[3] - 0.05, params[36] + mass[3] + 0.05)
            m.body_mass[6] = np.random.uniform(params[37] + mass[4] - 0.05, params[37] + mass[4] + 0.05)
            m.body_mass[7] = np.random.uniform(params[38] + mass[5] -0.05, params[38] + mass[5] + 0.05)

            # rope mass
            m.body_mass[-26:] = np.random.uniform(0.00001,0.00002)
            # rope stiffness
            m.jnt_stiffness[-50:] = np.random.uniform(0.0005, 0.005)
            # rope damping
            m.dof_damping[-50:] = np.random.uniform(0.0001, 0.0004)
            # ball size
            m.geom_size[23][0] = np.random.uniform(0.0095,0.011)
        else:
            m.dof_damping[[0,2]] = mp.x8_damping * scaling[0]
            m.dof_armature[[0,2]] = mp.x8_armature  * scaling[1]
            m.dof_frictionloss[[0,2]] = mp.x8_frictionloss * scaling[2]
            m.actuator_biasprm[[0, 2], 0] = mp.x8_maxT * scaling[3]
            m.actuator_biasprm[[0, 2], 1] = mp.x8_ramp * scaling[4]

            m.dof_damping[1] = mp.x8_16_damping * scaling[5]
            m.dof_armature[1] = mp.x8_16_armature * scaling[6]
            m.dof_frictionloss[1] = mp.x8_16_frictionloss * scaling[7]
            m.actuator_biasprm[1, 0] = mp.x8_16_maxT * scaling[8]
            m.actuator_biasprm[1, 1] = mp.x8_16_ramp * scaling[9]

            m.dof_damping[3:7] = mp.x5_damping * scaling[10]
            m.dof_armature[3:7] = mp.x5_armature * scaling[11]
            m.dof_frictionloss[3:7] = mp.x5_frictionloss * scaling[12]
            m.actuator_biasprm[3:7, 0] = mp.x5_maxT * scaling[13]
            m.actuator_biasprm[3:7, 1] = mp.x5_ramp * scaling[14]

            params = np.array(params)
            com = np.array(com)

            # CoM
            m.body_ipos[2, 0:3] = params[15:18] + com[0:3]
            m.body_ipos[3, 0:3] = params[18:21] + com[3:6]
            m.body_ipos[4, 0:3] = params[21:24] + com[6:9]
            m.body_ipos[5, 0:3] = params[24:27] + com[9:12]
            m.body_ipos[6, 0:3] = params[27:30] + com[12:15]
            m.body_ipos[7, 0:3] = params[30:33] + com[15:18]

            # Mass
            m.body_mass[2] = params[33] + mass[0]
            m.body_mass[3] = params[34] + mass[1]
            m.body_mass[4] = params[35] + mass[2]
            m.body_mass[5] = params[36] + mass[3]
            m.body_mass[6] = params[37] + mass[4]
            m.body_mass[7] = params[38] + mass[5]
            #print(m.body_mass)

    def load_ctrl_params(self, fn):
        # MAYBE use a sys id to find a better ctrl param?
        pass

    def step_wrapper(self, step_fn, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return step_fn(action)

    @classmethod
    def create_action_space(cls, action_space, low=[-0.49,-0.20,0.1,0,-1,0,0,-0.6], high=[-0.46,-0.19,0.2,0,-1,0,0,-0.3]):
        if action_space == 'eepose':
            # -0.4557282031,, -0.1982835829,, 0.0966898352
             # [-0.46, -0.175, 0.109]
            # quaternion -> [x, y, z, q0, q1, q2, q3]
            _high = np.array(high)
            _low = np.array(low)

            # _high = np.array([-0.46,-0.19,0.2,0,-1,0,0,-0.3])
            # _low = np.array([-0.49,-0.20,0.1,0,-1,0,0,-0.6])
        elif action_space == 'jointpos':
            _high = np.array([-1.4,2.5,2.3,0.7,2.5,1,-0.1])
            _low = np.array([-3.3,0.9,0.5,-1.2,0.1,-0.6,-0.6])
        elif action_space == 'jointvel':
            _high = np.array([1.5] * 7)
            _low = -_high
        elif action_space == 'eepose-delta':
            # _high = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.1, 0.05])
            # _low = np.array([-0.02,-0.02,-0.02,-0.1,-0.1,-0.1,-0.1,-0.05])
            _high = np.array([0.02, 0.02, 0.02,0,0,0,0, 0.5])
            _low = np.array([-0.02,-0.02,-0.02,0,0,0,0,-0.5])

            # _high = np.array([0.02, 0.02, 0.02,0.1,0.1,0.1,0.1, 0.5])
            # _low = np.array([-0.02,-0.02,-0.02,-0.1,-0.1,-0.1,-0.1,-0.5])

            # _high = np.array([0.1, 0.1, 0.1,0,0,0,0, 0.5])
            # _low = np.array([-0.1,-0.1,-0.1,0,0,0,0,-0.5])
        elif action_space == 'jointpos-delta':
            _high = np.array([0.1] * 7)
            _low = np.array([-0.1] * 7)
        elif action_space == 'eepose-euler':
            # [alpha, beta, gamma] rotation angle w.r.t [x,y,z]
            _high = np.array([-0.1,-0.05,0.4,np.pi,np.pi,np.pi,-0.1])
            _low = np.array([-0.65,-0.4,0.03,-np.pi,-np.pi,-np.pi,-0.6])
        elif action_space == 'eepose-euler-delta':
            raise NotImplementedError # TODO
        elif action_space == 'eepose-euler-vel':
            _high = np.array([0.1, 0.1, 0.1, np.pi/2, np.pi/2, np.pi/2, np.pi/4])
            _low = -_high
        elif action_space == 'xyz':
            _high = np.array([-0.1,-0.05,0.4])
            _low = np.array([-0.65,-0.4,0.03])
        elif action_space == 'xyz-delta':
            _high = np.array([0.02, 0.02, 0.02])
            _low = np.array([-0.02, -0.02, -0.02])
        elif action_space == 'xyz-vel':
            _high = np.array([0.1, 0.1, 0.1])
            _low = np.array([-0.1, -0.1, -0.1])
        else:
            raise NotImplementedError("Cannot recognize the given action space")
        return spaces.Box(_low.astype(np.float32), _high.astype(np.float32))

    def _set_action_space(self):
        # TODO verify & tailor the action space limits
        if self.config['action_space'] == 'eepose':
            step_fn = self.step_eepose
        elif self.config['action_space'] == 'jointpos':
            step_fn = self.step_jointpos
        elif self.config['action_space'] == 'jointvel':
            step_fn = self.step_jointvel
        elif self.config['action_space'] == 'eepose-delta':
            step_fn = self.step_eepose_delta
        elif self.config['action_space'] == 'jointpos-delta':
            step_fn = self.step_jointpos_delta
        elif self.config['action_space'] == 'eepose-euler':
            step_fn = self.step_eepose_euler
        elif self.config['action_space'] == 'eepose-euler-delta':
            raise NotImplementedError # TODO
        elif self.config['action_space'] == 'eepose-euler-vel':
            step_fn = self.step_eepose_euler_vel
        elif self.config['action_space'] == 'xyz':
            step_fn = self.step_xyz
        elif self.config['action_space'] == 'xyz-delta':
            step_fn = self.step_xyz_delta
        elif self.config['action_space'] == 'xyz-vel':
            step_fn = self.step_xyz_vel
        else:
            raise NotImplementedError("Cannot recognize the given action space")
        self.step = partial(self.step_wrapper, step_fn)
        self.action_space = TychoEnv.create_action_space(self.config['action_space'],\
            low=self.config['action_low'], high=self.config['action_high'])

    def _set_observation_space(self):
        if not self.config['state_space'] in \
            ['eepose', 'jointpos','eepose-obj','jointpos-obj']:
            raise NotImplementedError("Cannot recognize the given state space")
        self._use_eepose_state = 'eepose' in self.config['state_space']
        self._use_obj_state = 'obj' in self.config['state_space']
        self.state_size = 0
        self.state_size += 8 if self._use_eepose_state else 7
        self.state_size += 3 if self._use_obj_state else 0
        self.state_holder = np.zeros(self.state_size)
        state, _, _, _ = self.observe()
        self.observation_space = convert_observation_to_space(state)
        # TODO give a reasonable range so we can use a fixed normalization

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    @property
    def control_dt(self):
        return self.dt * self.control_repeat

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_task_reward(self):
        qpos = self.data.qpos.flat.copy()
        obj_pos = qpos[7:10]
        chop_tip = self.data.site_xpos[0]
        rot_chop_tip = self.data.site_xpos[3]
        mid_pos = (rot_chop_tip + chop_tip) / 2.0
        distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
        distance_to_goal = np.linalg.norm(obj_pos - self.goal)
        success = distance_to_obj <= 0.005 and distance_to_goal <= 0.02
        failure = distance_to_obj > 0.07
        ball_rad = self.config['ball_rad']
        if self.config['task'] in [Task.REACH, Task.GRASP]:
            reward = np.exp(-5 * distance_to_obj) + 0.3 * np.exp(-20 * distance_to_goal)
            if self.config['task'] == Task.GRASP:
                inter_tip_dist = np.linalg.norm(chop_tip - rot_chop_tip)
                success = success and np.abs(inter_tip_dist - 2 * ball_rad) <= ball_rad * 0.1
                reward += 0.15 * np.exp(-5 * np.abs(inter_tip_dist - 2 * ball_rad))
        elif self.config['task'] == Task.LIFT:
            inter_tip_dist = np.linalg.norm(chop_tip - rot_chop_tip)
            grasp_rew = -np.where(inter_tip_dist >= 2 * ball_rad, inter_tip_dist, 1.5 * inter_tip_dist)
            reward = (-5 * distance_to_obj) + (-10 * distance_to_goal) \
                     + grasp_rew
            success = success and np.abs(inter_tip_dist - 2 * ball_rad) <= ball_rad * 0.1
        else:
            print(f"Config: {self.config['task']}, Reach: {Task.REACH}")
            raise Exception(f"Unknown task {self.config['task']}")

        # if obj_pos[2] > 0.052:
        #     reward += 1000
        if (np.abs(obj_pos[2] - 0.05) < 0.01) or obj_pos[2] < 0.04:
            reward -= 5
        # hover reward
        if (np.abs(obj_pos[2] - 0.1) < 0.01):
            reward += 1
            if np.linalg.norm(obj_pos[2] - mid_pos[2]) < 0.0013:
                reward += 9
        if failure:
            reward -= 10

        return reward, success, False

    # ============================================
    # Core
    def observe(self):
        # return state, reward, done, info

        qpos = self.data.qpos.flat.copy()
        eepose = construct_choppose(self.ctrl.arm, qpos[0:7], target_at_ee=True)
        obj_pos = qpos[7:10]
        reward, success, failure = self._compute_task_reward()
        done = self._step_counter > self._max_episode_steps or failure
        state = self.gen_state(qpos[0:7], eepose, obj_pos)
        info = {
            'joint_pos': qpos[0:7],
            'obj_pos': obj_pos,
            'eepose':   eepose,
            'state':   state,
            'reward':  reward,
            'done':    done,
            'success': success
        }
        return state, reward, done, info

    def gen_state(self, jointpos, eepose, objpos):
        if self._use_eepose_state:
            self.state_holder[:8] = eepose
        else:
            self.state_holder[:7] = jointpos
        if self._use_obj_state:
            self.state_holder[-3:] = objpos
        return self.state_holder

    def sampleAction(self):
        if self.config['action_space'] == 'jointpos-delta':
            return self.action_space.sample()
        if self.config['action_space'] == 'jointpos':
            _high = np.array([0.1] * 7)
            _low = np.array([-0.1] * 7)
            action = spaces.Box(_low.astype(np.float32), _high.astype(np.float32)).sample()
            current_joint = self.data.qpos.flat.copy()[0:7]
            return current_joint + action
        if self.config['action_space'] == 'xyz-delta':
            return self.action_space.sample()
        if self.config['action_space'] == 'xyz':
            # sample xyz-delta
            _high = np.array([0.02, 0.02, 0.02])
            _low = np.array([-0.02, -0.02, -0.02])
            action = spaces.Box(_low.astype(np.float32), _high.astype(np.float32)).sample()
            current_xzy = self.data.qpos.flat.copy()[0:3]
            return action + current_xzy

        return self.action_space.sample()

    def dummy_step_jointpos_delta(self, target_joint_delta):
        # reset_eepos = self.config['reset_eepose_fn'](self, obj_pos)
        obj_pos = self.goal
        # joint_pos = construct_command(self.ctrl.arm, MOVING_POSITION, target_vector=reset_eepos)
        state = self.sim.get_state()
        state.qpos[0:7] = state.qpos[0:7] + target_joint_delta
        state.qpos[7:10] = obj_pos
        self.sim.set_state(state)
        self.dummy_step(timesteps=1)
        return self.observe()

    def dummy_step_xyz_delta(self, target_joint_delta):
        pass

    def reset(self, joint_pos=MOVING_POSITION):
        self.load_model_params(self.model_params_fn)
        self.ctrl.reset()
        self.sim.reset()
        self._step_counter = 0
        state = self.sim.get_state()
        if self.config['static_ball']:
            obj_pos = [-0.38, -0.26, 0.05] # SET a fixed obj pos
        else:
            obj_pos = randomize_pos_above_workspace([-0.38, -0.26,  0.05], [0.025, 0.01, 0])
            state.qvel[7:10] = (np.random.rand(3)-0.5)/10
        # import pdb;pdb.set_trace()
        reset_eepos = self.config['reset_eepose_fn'](self, obj_pos)
        perturb = np.random.uniform(-np.array([0.02,0.005,0.02,0,0,0,0,0]), np.array([0.02,0.005,0.02,0,0,0,0,0])) if self.dr else np.zeros(8)
        reset_eepos = np.array([-0.385, -0.28, 0.05001724,0,-1,0,0, -0.36]) + perturb
        # reset_eepos = np.array([-0.385, -0.28, 0.05001724,0, 0.9659258, 0, 0.258819, -0.36]) + perturb
        joint_pos = construct_command(self.ctrl.arm, joint_pos, target_vector=reset_eepos)
        state.qpos[0:7] = joint_pos
        state.qpos[7:10] = obj_pos
        self.goal = randomize_pos_above_workspace([-0.46, -0.17,  0.3678], [0.14] * 3)
        if self.config['task'] in [Task.REACH, Task.GRASP]:
            self.goal = state.qpos[7:10]
        else:
            self.goal = [-0.38, -0.26, 0.1]
            self.data.mocap_pos[1,:] = self.goal
        self.sim.set_state(state)
        for _ in range(1):
            self.sim.forward()
        # Warmup the PID controller
        _, _, _, observation = self.observe()
        # import pdb;pdb.set_trace()
        target = observation['joint_pos'][0:7]
        for _idx in range( int(1.0 / self.control_repeat) ):
            env.step_jointpos(target)
        return self.observe()[0]

    # for viz_expert
    def reset_ee_pose(self, ee_pose, joint_pos=MOVING_POSITION):
        assert(len(ee_pose) == 8)
        ee_xyz = ee_pose[0:3]
        ee_quat = scipyR.from_quat(ee_pose[3:7]).as_matrix()
        joint_pos[0:6] = self.ctrl.arm.get_IK(joint_pos, ee_xyz, ee_quat)
        joint_pos[6] = ee_pose[7]
        return self.reset(joint_pos)

    # for viz_expert
    def reset_m6(self, position):
        state = self.sim.get_state()
        state.qpos[7:10] = position
        self.sim.set_state(state)
        self.sim.forward()

    def _reset_jointpos(self, position=MOVING_POSITION):
        state = self.sim.get_state()
        state.qpos[0:7] = position
        self.sim.set_state(state)
        self.sim.forward()

    def _reset_ball_xyz(self, position): # NOT IN USE
        state = self.sim.get_state()
        state.qpos[7:10] = position
        self.sim.set_state(state)
        self.sim.forward()

    def pwm_to_torque(self, pwm, gravity=None):
        pwm = np.clip(pwm, -1, 1)
        maxtorque = np.array(self.model.actuator_biasprm[0:7, 0])
        speed_24v = np.array(self.model.actuator_biasprm[0:7, 1])
        qvel = np.array(self.data.qvel.flat[0:7])
        ctrl = np.multiply(pwm - np.divide(np.abs(qvel), speed_24v), maxtorque)
        # import pdb;pdb.set_trace()
        if gravity is not None:
            ctrl += gravity
        ctrl = np.clip(ctrl,-maxtorque,maxtorque)
        return ctrl


    def _step(self, repeat, target_pos=None, target_vel=None):
        #print('position qpos', self.data.qpos)
        #print('velocity qvel', self.data.qvel)
        #print('force qfrc_actuation', self.data.actuator_force)
        # grab states and pass to controller to generate PWM
        position = self.data.qpos.flat.copy()[0:7]
        velocity = self.data.qvel.flat.copy()[0:7]

        contact_force = self.data.actuator_force.flat.copy()[0:7] # TODO note that we only get "active force"?
        pwm = self.ctrl.act(target_pos, target_vel, position, velocity, contact_force)
        ctrl = self.pwm_to_torque(pwm,None)
        for _ in range(repeat):
            self.sim.data.ctrl[:] = ctrl
            #self.sim.data.qfrc_applied[:7] = ctrl
            #print(self.sim.data.qfrc_applied)
            self.sim.step()

    # does NOT call forward(), so do that manually (or call step())
    def set_target_indicator(self, xyz):
        state = self.sim.get_state()
        state.qpos[14:17] = xyz
        self.sim.set_state(state)

    def step_xyz(self, target_xyz):
        self.set_target_indicator(target_xyz)

        current_pos = self.data.qpos.flat.copy()[0:7]
        target_eepose = construct_choppose(self.ctrl.arm, MOVING_POSITION, target_at_ee=True)
        target_eepose[0:3] = target_xyz
        target_jointpos = construct_command(self.ctrl.arm, current_pos, target_vector=target_eepose)
        return self.step_jointpos(target_jointpos)

    def step_xyz_delta(self, target_xyz_delta):
        current_pos = self.data.qpos.flat.copy()[0:7]
        current_eepose = construct_choppose(self.ctrl.arm, current_pos[0:7], target_at_ee=True)
        current_xyz = current_eepose[0:3]
        target_eepose = construct_choppose(self.ctrl.arm, MOVING_POSITION, target_at_ee=True)
        target_eepose[0:3] = current_xyz + target_xyz_delta
        self.set_target_indicator(target_eepose[:3])
        target_jointpos = construct_command(self.ctrl.arm, current_pos, target_vector=target_eepose)
        return self.step_jointpos(target_jointpos)

    def step_xyz_vel(self, target_xyz_vel):
        cmd_vel = np.hstack((target_xyz_vel, (0,) * 4))
        return self.step_eepose_euler_vel(cmd_vel)

    def step_eepose_euler_vel(self, target_eepose_euler_vel):
        assert target_eepose_euler_vel.shape == (7,) # posVel, rotVel, chop
        current_jointpos = self.data.qpos.flat.copy()[:7]
        current_eepose = construct_choppose(self.ctrl.arm, current_jointpos, target_at_ee=True)[:-1]
        params = np.hstack((current_eepose[3:], current_eepose[:3])) # move xyz to the end
        trf = get_transformation_matrix_from_quat(params)
        cmd_vel = target_eepose_euler_vel[:-1]
        _, joint_vels = self.ctrl.arm.get_jog(current_jointpos, trf, cmd_vel, 0) # dt not used
        joint_vels = joint_vels.flatten()
        joint_vels = np.hstack((joint_vels, target_eepose_euler_vel[-1]))
        return self.step_jointvel(joint_vels)

    def step_eepose_delta(self, target_ee_delta):
        current_pos = self.data.qpos.flat.copy()[0:7]

        current_eepose = construct_choppose(self.ctrl.arm, current_pos[0:7], target_at_ee=True)


        target_eepose = current_eepose + target_ee_delta
        # import pdb;pdb.set_trace()
        target_eepose = np.clip(target_eepose, self.config['action_low'], self.config['action_high'])
        self.set_target_indicator(target_eepose[:3])
        target_jointpos = construct_command(self.ctrl.arm, current_pos, target_vector=target_eepose)
        return self.step_jointpos(target_jointpos)

    def step_jointpos_delta(self, target_joint_delta):
        current_pos = self.data.qpos.flat.copy()[0:7]
        return self.step_jointpos(current_pos + target_joint_delta)

    def step_eepose(self, target_eepose):
        self.set_target_indicator(target_eepose[:3])
        current_pos = self.data.qpos.flat.copy()[0:7]
        target_jointpos = construct_command(self.ctrl.arm, current_pos, target_vector=target_eepose)
        return self.step_jointpos(target_jointpos)

    def step_eepose_euler(self, target_eepose_euler):
        pos = target_eepose_euler[:3]
        self.set_target_indicator(pos)
        quat = euler_angles_to_quat(target_eepose_euler[3:6])
        open_angle = target_eepose_euler[6]
        target_eepose = np.hstack((pos, quat, open_angle))
        return self.step_eepose(target_eepose)

    def step_jointpos(self, target_jointpos):
        self._step_counter += 1
        for _ in range(self.control_repeat):
            self._step(self.frame_skip, target_pos=target_jointpos)
        return self.observe()

    def step_jointvel(self, target_jointvel):
        self._step_counter += 1
        for _ in range(self.control_repeat):
            self._step(self.frame_skip, target_vel=target_jointvel)
        return self.observe()

    def dummy_step(self, timesteps=100):
        self._step_counter += 1
        for _ in range(timesteps * self.frame_skip * self.control_repeat):
            self.sim.data.ctrl[:] = 0
            self.sim.step()

    # ============================================
    # Viewer & rendering, copy from mujoco_py
    def render(self,
               mode='human',
               width=720,
               height=480,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'depth_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array' or mode == 'depth_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # ============================================
    # Depre, maybe use it in future
    def _set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

if __name__ == '__main__':

    env = TychoEnv(onlyPosition=False)

    from timeit import default_timer as timer
    print(env.reset())
    observation = env.reset()
    env.render()
    print('dt', env.dt)
    print('control dt', env.control_dt)
    start = timer()
    act = observation[0:8].copy()
    # act = [-0.46, -0.195, 0.1, 0,-1,0,0, -0.3761813528]
    print('About to send EE target', act)
    for _idx in range(1000000):
        #print(_idx)
        # if _idx % 10 == 0:
        #     target[0:3] += np.array([0,0,0.05])
        # import pdb;pdb.set_trace()
        new_obs, reward, done, info = env.step(act) # keep the pose
        # print(new_obs)
        print(reward)
        # act[0] -=0.01
        #print('norm of distance to the desired static target', np.linalg.norm(new_obs['state'][0:3] - observation['state'][0:3]))
        #print('diff btw joints to the desired static target', new_obs['joint_pos'] - observation['joint_pos'])
        env.render()
    #print('Ending state', env.step(observation['state'][0:8]))
    env.render()
    end = timer()
    print('Time Elapsed (seconds)', end - start)
