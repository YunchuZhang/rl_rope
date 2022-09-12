# test
import gym
import time
import numpy as np
import pickle
import torch
import argparse
from tycho_env import TychoEnv# prepare environment
from tycho_env import TychoEnv_rope# prepare environment
parser = argparse.ArgumentParser(
    description='test sac')
parser.add_argument('--load', '-l', type=str,
                    required=False, help='location to restore policy')
parser.add_argument('--save_traj', '-s', type=bool,default=False,
                    required=False, help='save trajectory for RL training')
args, overrides = parser.parse_known_args()



reach, grasp_cnt = False, 0
def hack_controller_20hz(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    obj_pos[2] -=0.01
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] -= 0.0095
    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.0012 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        # print(dist_z)
    elif cnt > 6  and distance_to_goal > 0.01 and grasp_cnt < 8:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        act[1] += 0.003
        act[-1] = -0.5
        act[2] += 0.013
        grasp_cnt += 1
    elif grasp_cnt >= 8 and distance_to_goal > 0.005:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.02
        # print(act)
    return act
def hack_controller_30deg_20hz(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    obj_pos[2] -=0.01
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    # mid_pos[0] += 0.025
    # mid_pos[1] += 0.005
    # mid_pos[2] -= 0.0095

    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] += 0.0012

    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0
    if dist_z > 0.002 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        # act[1] -= 0.01
        act[2] += 0.012
        print(dist_z)
    elif cnt > 6  and distance_to_goal > 0.01 and grasp_cnt < 8:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (2)
        act[1] += 0.003
        act[-1] = -0.5
        act[2] += 0.013
        grasp_cnt += 1
    elif grasp_cnt >= 8 and distance_to_goal > 0.005:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.02
        # print(act)
    return act
def hack_controller(env,cnt):
    global reach, grasp_cnt
    qpos = env.data.qpos.flat.copy()
    obj_pos = qpos[7:10]
    chop_tip = env.data.site_xpos[0]
    rot_chop_tip = env.data.site_xpos[3]
    mid_pos = (rot_chop_tip + chop_tip) / 2.0
    mid_pos[0] += 0.025
    mid_pos[1] += 0.005
    mid_pos[2] -= 0.0095
    distance_to_obj = np.linalg.norm(obj_pos - mid_pos)
    distance_to_goal = np.linalg.norm(obj_pos - env.goal)

    dist_z = np.linalg.norm(obj_pos[2] - mid_pos[2])

    act = np.zeros(8,dtype=np.float32)
    act[-1] = 0.0

    if dist_z > 0.012 and not reach:
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1)
        act[2] += 0.007

    elif cnt > 100 and distance_to_goal > 0.01 and grasp_cnt < 75:
        # print(grasp_cnt)
        reach = True
        for i in range(3):
            act[i] = (obj_pos-mid_pos)[i] * (1)
        act[-1] = -0.1
        act[2] += 0.007
        grasp_cnt += 1
    elif grasp_cnt >= 75 and distance_to_goal > 0.01:
        for i in range(3):
            act[i] = (env.goal-obj_pos)[i] * (1)
        act[-1] = -0.01
        act[2] += 0.005

    return act

def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    # action = np.clip(action, action_space.low, action_space.high)
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low + 1e-8)) - 1.0


def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low + 1e-8))

def evaluate_on_environment(
    env: gym.Env, algo, n_trials: int = 10, render: bool = False):
    """Returns scorer function of evaluation on environment.
    """
    global reach, grasp_cnt
    episodes = []
    episode_rewards = []
    for _ in range(n_trials):
        reach, grasp_cnt = False, 0
        episode = {}
        obs = []
        r = []
        d = []
        act = []
        observation = env.reset()
        episode_reward = 0.0
        cnt = 0
        while True:

            cnt += 1
            # take action
            action = algo.predict([observation])
            action = unscale_action(env.action_space, action)
            # action = env.action_space.sample()

            # action = hack_controller(env,cnt)
            #action = hack_controller_20hz(env,cnt)
            obs.append(observation.copy())
            act.append(action.copy())

            observation, reward, done, _ = env.step(action)
            # print(observation)
            # print(env.sim.get_state().qvel[7:10])
            print(action)
            r.append(reward.copy())
            d.append(done)

            episode_reward += reward
            if render:
                env.render()


            if done:
                break

        episode = {"obs":np.array(obs,dtype=np.float32),"act":np.array(act,dtype=np.float32),"rew":np.array(r,dtype=np.float32),"done":np.array(d,dtype=np.float32)}
        print(cnt)
        if cnt > 60:
            episodes.append(episode)
            episode_rewards.append(episode_reward)
    if args.save_traj:
        np.save('demo_rope_random_20hz',np.asarray(episodes))
    print(len(episode_rewards))
    return float(np.mean(episode_rewards))

class D3Agent():
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def load(self, model_folder, device):
        # load is handled at init
        pass
    # For 1-batch query only!
    def predict(self, sample):

        with torch.no_grad():
            input = torch.from_numpy(sample[0]).float().unsqueeze(0).to('cuda:0')
            at = self.policy(input)[0].to('cpu').detach().numpy()
        return at
import d3rlpy
torch.manual_seed(123)
d3rlpy.seed(123)

device='cuda:0'
policy = torch.jit.load(args.load)
policy.to(device)
agent = D3Agent(policy, device)
# env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":True,
#             'action_low': [-0.40,-0.30,0.04,0, 0.9659258, 0, 0.258819,-0.57],
#             'action_high':[-0.37,-0.27,0.11,0, 0.9659258, 0, 0.258819,-0.2],
#     })
# env = TychoEnv_rope(onlyPosition=False,config={"state_space":"eepose-obj-vel", "action_space":"eepose-delta","static_ball":False,
#             'action_low': [-0.5,-0.5,0.03,0,-1,0,0,-0.57],
#             'action_high':[0.,-0.1,1.,0,-1,0,0,-0.2],
#             "dr":False,
#     })
# no rotation restriction
# env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
#             'action_low': [-0.40,-0.30,0.04,-1,-1,-1,-1,-0.57],
#             'action_high':[-0.37,-0.27,0.11,1,1,1,1,-0.2],
#     })
env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
            'action_low': [-0.40,-0.30,0.04,0,-1,0,0,-0.57],
            'action_high':[-0.37,-0.27,0.11,0,-1,0,0,-0.2],
    })
env.seed(123)



print(evaluate_on_environment(env,agent,n_trials=10,render=True))
