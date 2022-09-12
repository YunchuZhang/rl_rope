from .Planner import RRT
from .TychoEnv import TychoEnv

class TychoRRT(RRT):
    def __init__(self, max_step=1<<20, goal_sample=0.05):
        action_space = TychoEnv.create_action_space("jointpos")
        super().__init__(sample_func=action_space.sample, dim=action_space.shape[0],
                         max_step=max_step, goal_sample=goal_sample)
