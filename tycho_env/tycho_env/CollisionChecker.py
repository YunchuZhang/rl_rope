import numpy as np

from tycho_env.utils import DH_params, get_DH_transformation

# ----------------------------------------------------

class CollisionChecker():
    THRESHOLD = 0.005
    SPHERES_CONFIG = [
    # Link, transform, radius
     [0, [-0.02,0,0.02], 0.1],
     [2, [0.02,0,0], 0.105],
     [2, [0.15,0,0.02], 0.04],
     [2, [0.22,0,0.02], 0.04],
     [2, [0.305,0,-0.02], 0.075],
     [3, [0.07,0,0.02], 0.035],
     [3, [0.131,0,0.02], 0.035],
     [3, [0.1845,0,0.02], 0.035],
     [3, [0.2415,0,0.02], 0.035],
     [3, [0.30575,0,-0.01125], 0.061],
     [4, [-0.01,-0.055,-0.01], 0.0555],
     [4, [-0.04,-0.055,-0.01], 0.0455],
     [5, [ 0.01,-0.05,0], 0.05],
     [5, [ -0.03875,-0.055,-0.00125], 0.0425],
     [6, [ 0.0, 0.04375,-0.0025], 0.05],
     [6, [ -0.045, 0.06, 0.0025], 0.04],
     # Chop ball
     [6, [ 0.118368, 0.07914739999999999, 0.024354999999999995], 0.0025],
     [6, [ -0.1181944999999989, 0.07914739999999999, 0.024354999999999995], 0.003],
     [7, [0.11593750000000001, 0.0525, 0.023125], 0.0025],
     [7, [-0.09600699999999887, -0.04218749999999999, 0.0225], 0.00375]
    ]
    Z_MIN = 0.032
    Z_MAX = 0.6
    X_MAX = 0.2

    def __init__(self):
        NUM_SPHERES = len(self.SPHERES_CONFIG)
        self.SPHERE_JOINTS = np.array([s[0] for s in self.SPHERES_CONFIG])
        self.SPHERE_SIZES = np.array([s[-1] for s in self.SPHERES_CONFIG])
        self.BALLA, self.BALLB, self.BALLDis = [], [], []
        for i in range(NUM_SPHERES-2):
            for j in range(i+2, NUM_SPHERES):
                if self.SPHERE_JOINTS[j] > self.SPHERE_JOINTS[i] + 1:
                    self.BALLA.append(i)
                    self.BALLB.append(j)
                    self.BALLDis.append(self.SPHERE_SIZES[i] + self.SPHERE_SIZES[j])
        self.BALLDis = np.array(self.BALLDis)

        self.SPHERE_TRANFORM = np.ones((NUM_SPHERES, 4, 1))
        self.SPHERE_TRANFORM[:,0:3, 0] = [s[1] for s in self.SPHERES_CONFIG]

    def is_collision(self, joint_position):
        joint_loc = np.zeros((8, 4, 4))
        joint_loc[0] = np.eye(4)
        for i, ((alpha, a, d), theta) in enumerate(zip(DH_params, joint_position)):
            joint_loc[i+1] = joint_loc[i].dot(get_DH_transformation(alpha, a, theta, d))
        sphere_locs = np.einsum('ijk,ikl->ijl', joint_loc[self.SPHERE_JOINTS], self.SPHERE_TRANFORM)[:,0:3].reshape(-1,3)
        # ball x ball
        pair_dis = np.linalg.norm(sphere_locs[self.BALLA] - sphere_locs[self.BALLB], axis=1)
        if np.any(pair_dis < self.BALLDis + self.THRESHOLD):
            return True
        # ball x line
        dist_to_chop1 = lineseg_dist(sphere_locs[:-6,], sphere_locs[-1], sphere_locs[-2])
        dist_to_chop2 = lineseg_dist(sphere_locs[:-6,], sphere_locs[-3], sphere_locs[-4])
        if np.any(dist_to_chop1 < self.SPHERE_SIZES[:-6] + self.THRESHOLD) or np.any(dist_to_chop2 < self.SPHERE_SIZES[:-6] + self.THRESHOLD):
            return True

    def grab_ball_distance(self, joint_position):
        joint_loc = np.zeros((8, 4, 4))
        joint_loc[0] = np.eye(4)
        for i, ((alpha, a, d), theta) in enumerate(zip(DH_params, joint_position)):
            joint_loc[i+1] = joint_loc[i].dot(get_DH_transformation(alpha, a, theta, d))
        sphere_locs = np.einsum('ijk,ikl->ijl', joint_loc[self.SPHERE_JOINTS], self.SPHERE_TRANFORM)[:,0:3].reshape(-1,3)
        # ball x ball
        pair_dis = np.linalg.norm(sphere_locs[self.BALLA] - sphere_locs[self.BALLB], axis=1)
        return pair_dis

def lineseg_dist(points, lineA, lineB):
    distances = np.ones(len(points))
    line = np.divide(lineB - lineA, np.linalg.norm(lineB - lineA))
    PA = lineA - points
    BP = points - lineB
    s = np.dot(PA, line)
    t = np.dot(BP, line)
    inbtw_idx = np.logical_and(s < 0, t < 0) # Tries to tell if P lies btw A and B
    if np.any(inbtw_idx):
        # Length of cross product equals to the area
        distances[inbtw_idx] = np.linalg.norm(np.cross(points[inbtw_idx] - lineA, line), axis=1)
    distances[~inbtw_idx] = np.minimum(np.linalg.norm(PA[~inbtw_idx], axis=1), np.linalg.norm(BP[~inbtw_idx],axis=1))
    return distances


def time_implementation(env, cc, repeat):
    import timeit
    def timing_collision_checker():
        jp = env.action_space.sample()
        return cc.is_collision(jp)
    print(f"Avg execution time is: {timeit.timeit(stmt = timing_collision_checker, number = repeat)/repeat}")

def verify_implementation(env, cc, repeat):

    def get_sphere_loc(joint_position, sphere_config):
        joint_id = sphere_config[0]
        last_transformation = np.ones(4)
        last_transformation[0:3] = sphere_config[1]

        ee = np.eye(4)
        for (alpha, a, d), theta in zip(DH_params[:joint_id,:], joint_position[:joint_id]):
          ee = ee.dot(get_DH_transformation(alpha, a, theta, d))
        ee = ee.dot(last_transformation.reshape(4,1))
        return ee[0:3].reshape(3)

    def get_distances(joint_positions):
        num_spheres = len(cc.SPHERES_CONFIG)
        sphere_sizes = cc.SPHERE_SIZES
        sphere_locs = np.zeros((num_spheres,3))
        for i, s in enumerate(cc.SPHERES_CONFIG):
            sphere_locs[i,:] = get_sphere_loc(joint_positions, s)
        distances = []
        for i in range(num_spheres-2):
            for j in range(i+2, num_spheres):
                if cc.SPHERES_CONFIG[j][0] <= cc.SPHERES_CONFIG[i][0] + 1:
                    continue # Not checking collision for neighboring links
                distances.append(np.linalg.norm(sphere_locs[i] - sphere_locs[j]))
        return np.array(distances)

    for _ in range(repeat):
        jp = env.action_space.sample()
        a = cc.grab_ball_distance(jp)
        b = get_distances(jp)
        assert np.all(np.isclose(a, b))

    print("\nFaster implementation yielded the same ball distance with the slow implementation")

# -------------------------------------

if __name__ == '__main__':
    from tycho_env import TychoEnv
    env = TychoEnv(config={"action_space":"jointpos"})
    cc = CollisionChecker()
    verify_implementation(env, cc, 100)
    time_implementation(env, cc, 10000)
