import numpy as np
import math

from .CollisionChecker import CollisionChecker

def compute_path_len(path):
  a = np.array(path[:-1])
  b = np.array(path[1:])
  return np.sum(np.linalg.norm(b-a, axis=1))

class Planner:
  COLLISION_STEPSIZE = 0.005
  CLAMP = np.array([0.1, 0.15, 0.2, 0.35, 0.5, 0.5, 0.5])
  CLAMP_MIN = np.array([-0.1, -0.15, -0.2, -0.35, -0.5, -0.5, -0.5])
  def __init__(self):
    self.cc = CollisionChecker()

  def extend(self, start, target):
    diff = target - start
    ratio = np.divide(np.abs(diff), self.CLAMP)
    if max(ratio) <= 1:
      return target
    return start + diff / max(ratio)
    #diff = np.clip(target - start, self.CLAMP_MIN, self.CLAMP)
    #return start + diff

  def unreachable(self, start, target):
    diff = target - start
    length = math.ceil(np.linalg.norm(diff) / self.COLLISION_STEPSIZE)
    step = diff / np.linalg.norm(diff) * self.COLLISION_STEPSIZE
    vex = np.array(start)
    for _c in range(length-1):
      vex += step
      if self.cc.is_collision(vex):
        return True, vex - step, _c
    return False, target, length

  def smooth(self, path, max_iter=100):
    for i in range(len(path)-1): assert not self.unreachable(path[i], path[i+1])[0]

    for i in range(max_iter):
      new_vex = self.sample_func()
      if self.cc.is_collision(new_vex): continue
      s, t = -1, len(path)
      is_unreacable = True
      while is_unreacable and s + 1 < len(path):
        s += 1
        is_unreacable, _, _ = self.unreachable(path[s], new_vex)
      is_unreacable = True
      while is_unreacable and s + 2 < t:
        t -= 1
        is_unreacable, _, _ = self.unreachable(path[t], new_vex)
      if s+2 < t:
        _len = compute_path_len(path[s:t+1])
        new_sec = [path[s], new_vex, path[t]]
        if _len > compute_path_len(new_sec):
          path = path[:s+1] + [new_vex] + path[t:]

    i = 0
    while i + 2 < len(path):
      j = len(path) - 1
      while j > i + 1 and self.unreachable(path[i], path[j])[0]:
        j -= 1
      if j > i+1:
        path = path[:i+1] + path[j:]
      i += 1

    interpolated_path = [path[0]]
    for i in range(1,len(path)):
      reach_vex = path[i-1]
      while not np.all(np.isclose(reach_vex, path[i])):
        reach_vex = self.extend(reach_vex, path[i])
        interpolated_path.append(reach_vex)
    return interpolated_path


class RRT(Planner):
  def __init__(self, sample_func, max_step=1<<20, goal_sample=0.05, dim=7):
    super().__init__()
    self.max_step = max_step
    self.goal_sample = goal_sample
    self.graph = np.zeros((max_step, dim))
    self.father = np.zeros(max_step, dtype=int)
    self.sample_func = sample_func

  def find_closet(self, vex, valid_idxes):
    distances = np.linalg.norm(self.graph[:valid_idxes] - vex, axis=1)
    idx = np.argmin(distances)
    return idx, self.graph[idx]

  def plan(self, start_vex, end_vex):
    self.graph[0][:] = np.array(start_vex)
    end_vex = np.array(end_vex)
    self.father[0] = -1
    i = 1
    while i < self.max_step:
      rand_vex = self.sample_func() if np.random.rand() > self.goal_sample else end_vex
      if self.cc.is_collision(rand_vex): continue
      closet_idx, cloest_vex = self.find_closet(rand_vex, i)
      rand_vex = self.extend(cloest_vex, rand_vex)
      is_unreacable, new_vex, cost = self.unreachable(cloest_vex, rand_vex)
      if cost > 0:
        self.graph[i][:] = new_vex
        self.father[i] = closet_idx
        is_unreacable, _, _ = self.unreachable(new_vex, end_vex)
        if is_unreacable is False:
          self.size = i
          return self.generate(i) + [end_vex]
      i += 1
    return None

  def generate(self, idx):
    trajectory = []
    while idx >= 0:
      print(idx, self.graph[idx])
      trajectory.append(self.graph[idx])
      idx = self.father[idx]
    trajectory.reverse()
    return trajectory

if __name__ == '__main__':

  # ===================================================
  # 2D DUMMY TEST

  if True:
    import matplotlib.pyplot as plt
    obstacleList = [ # [x,y,size]
    (4,0,1.2),(4,2,1.2),(4,4,1.2),(4,6,1.2),(4,8,1.2),(4,10,1.2),(4,12,1.2),
    (8,14,1.2),(8,4,1.2),(8,6,1.2),(8,8,1.2),(8,10,1.2),(8,12,1.2),
    (12,8,0.9),(14,8.5,1.2),
          (5, 5, 0.5),
          (3.2, 6, 1),
          (2.6, 8.9, 1),
          (3, 10.3, 1),
          (6.6, 3.9, 1),
          (9.4, 4.52, 1),
      ]
    circles = [plt.Circle((ox, oy), size, color='black') for ox, oy, size in obstacleList]
    for c in circles:
      plt.gca().add_patch(c)
    plt.xlim([0,15])
    plt.ylim([0,15])
    plt.gca().set_box_aspect(1)

    def dummy_collision_checker(point):
      x,y = point
      for (ox, oy, size) in obstacleList:
        if (x - ox) ** 2 + (y - oy) ** 2 < size ** 2:
          return True
      return False
    def dummy_sampler():
      return np.array([np.random.uniform(0, 15), np.random.uniform(0, 15)])
    dummy_clamp = np.array([1.0, 1.0])
    rrt_planner = RRT(dummy_sampler, dim=2)
    rrt_planner.cc.is_collision = dummy_collision_checker
    rrt_planner.CLAMP = dummy_clamp
    rrt_planner.CLAMP_MIN = -dummy_clamp
    path = rrt_planner.plan(np.array([0.,0.]), np.array([15.,15.]))

    for idx, (x,y) in enumerate(rrt_planner.graph[:rrt_planner.size]):
      f = rrt_planner.father[idx]
      a,b = rrt_planner.graph[f]
      plt.plot([a,x],[b,y],'r-')

    print('Before smoothing', compute_path_len(path))
    print('Vex collision?', np.any([rrt_planner.cc.is_collision(vex) for vex in path]))
    print('Asserting edge non-collision')
    for i in range(len(path)-1): assert not rrt_planner.unreachable(path[i], path[i+1])[0]

    print('Apply smoothing')
    path = rrt_planner.smooth(path, max_iter=100)
    print('After smoothing', compute_path_len(path))
    x, y = [], []
    for _x, _y in path:
      x.append(_x)
      y.append(_y)
    plt.plot(x,y,'b-o', alpha=0.5)
    for vex in path:
      print(vex, rrt_planner.cc.is_collision(vex))
    plt.show()

  # ===================================================
  # Pybullet robot testing
  #
  import pybullet as p
  import time
  from viz_collision_balls import Bullet, URDF_PATH
  from tycho_env import TychoEnv
  from tycho_env.utils import MOVING_POSITION

  my_bullet = Bullet(gui=True)
  env = TychoEnv(config={"action_space":"jointpos"})
  rrt_planner = RRT(env.action_space.sample, dim=7)
  my_bullet.load_robot(URDF_PATH)
  my_bullet.marionette(MOVING_POSITION)

  cursor = 0

  keys = ''
  smoothed_path = []
  while True:
      keys = p.getKeyboardEvents()
      p.stepSimulation()
      time.sleep(0.01)

      test_keys = ['q','p','n','s']
      pressed_key = None
      for _k in test_keys:
          if ord(_k) in keys:
              state = keys[ord(_k)]
              if (state & p.KEY_WAS_RELEASED):
                  pressed_key = _k
                  break

      if pressed_key == None:
          continue
      elif pressed_key == 'q':
          break
      elif pressed_key == 'p':
        jp = my_bullet.get_joint_positions()
        print(f"Planning from {jp} to home")
        path = rrt_planner.plan(np.array(jp), MOVING_POSITION)
        print(f"Found a path of length {compute_path_len(path)}, unsmooth")
        smoothed_path = rrt_planner.smooth(path, max_iter=10)
        print(f"Found a path of length {compute_path_len(smoothed_path)}, smooth")
        print(smoothed_path)
        print(f"Press n to view the intermediate joint positions")
      elif pressed_key == 'n':
        if len(smoothed_path):
          my_bullet.marionette(smoothed_path.pop(0))
      elif pressed_key == 's':
        jp = env.action_space.sample()
        my_bullet.marionette(jp)
