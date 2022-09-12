import d3rlpy
import gym
import os
from tycho_env import TychoEnv, TychoEnv_rope# prepare environment
import argparse
parser = argparse.ArgumentParser(
    description='train sac')
parser.add_argument('--output', '-o', type=str,
                    required=True, help='location to store results')
parser.add_argument('--load', '-l', type=str,
                    required=False, help='location to restore weights')
parser.add_argument('--load_demo', '-d', type=str,
                    required=False, help='load saved trajectory for RL training')
args, overrides = parser.parse_known_args()
OUT_DIR = args.output
MODEL = args.load
DEMO = args.load_demo
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

if not os.path.exists(OUT_DIR+'/logs'):
    os.mkdir(OUT_DIR+'/logs')


# env = TychoEnv(onlyPosition=False)
# eval_env = TychoEnv(onlyPosition=False)
# env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":True})
# eval_env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":True})

# rope stop env
env = TychoEnv_rope(onlyPosition=False,config={"state_space":"eepose-obj-vel", "action_space":"eepose-delta","static_ball":False,
            'action_low': [-0.5,-0.5,0.03,0,-1,0,0,-0.57],
            'action_high':[0.,-0.1,1.,0,-1,0,0,-0.2],
            "dr":False,
    })
eval_env = TychoEnv_rope(onlyPosition=False,config={"state_space":"eepose-obj-vel", "action_space":"eepose-delta","static_ball":False,
            'action_low': [-0.5,-0.5,0.03,0,-1,0,0,-0.57],
            'action_high':[0.,-0.1,1.,0,-1,0,0,-0.2],
            "dr":False,
    })
# torch.manual_seed(123)
#30deg
# env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
#             'action_low': [-0.40,-0.30,0.04,0, 0.9659258, 0, 0.258819,-0.57],
#             'action_high':[-0.37,-0.27,0.11,0, 0.9659258, 0, 0.258819,-0.2],
#     })
# eval_env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
#             'action_low': [-0.40,-0.30,0.04,0, 0.9659258, 0, 0.258819,-0.57],
#             'action_high':[-0.37,-0.27,0.11,0, 0.9659258, 0, 0.258819,-0.2],
#     })
# no rotation restriction
# env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
#             'action_low': [-0.40,-0.30,0.04,-1,-1,-1,-1,-0.57],
#             'action_high':[-0.37,-0.27,0.11,1,1,1,1,-0.2],
#     })
# eval_env = TychoEnv(onlyPosition=False,config={"action_space":"eepose-delta","static_ball":False,"dr":False,
#             'action_low': [-0.40,-0.30,0.04,-1,-1,-1,-1,-0.57],
#             'action_high':[-0.37,-0.27,0.11,1,1,1,1,-0.2],
#     })

d3rlpy.seed(123)
env.seed(123)
eval_env.seed(123)

sac = d3rlpy.algos.SAC(use_gpu=True,actor_learning_rate= 3e-4,
        critic_learning_rate= 3e-4,
        temp_learning_rate= 3e-4,)
if args.load is not None:
	import pdb;pdb.set_trace()
	print("loding  ",MODEL)
	sac.build_with_env(env)
	sac.load_model(MODEL)

# prepare replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)
# start training
sac.fit_online(env, buffer, n_steps=5000000, random_steps = 5000, \
	logdir=OUT_DIR,eval_env=eval_env, tensorboard_dir=OUT_DIR+'/logs',save_interval=10,load_demo=DEMO)
sac.save_policy(os.path.join(OUT_DIR,'policy.pt'))
sac.save_model(os.path.join(OUT_DIR,'model.pt'))
