import numpy as np
import hebi

from .utils import get_hrdf_path
from .custom_fk import create_custom_robot_model

class ArmContainer(object):
  # The Arm Container maintains
  # - group         talks to the physical robot and can send command
  # - _robot_full   kinematic robot model with 7 joints, used for gravity comp
  # - _robot_ee     kinematic robot model with 6 joints, used for FK and IK.
  #                 ee stands for end_effector. The end effector position of
  #                 the kinematic model is (as of 2021.6) defined to be the
  #                 tip of the static chopsticks. Which in this round of
  #                 calibration is 10 cm from the side of the mounting bracket.

  def __init__(self, group, model_full, model_ee, module_name):
    self.group = group
    self._robot_full = model_full
    self._robot_ee = model_ee
    self._masses = model_full.masses
    self.module_name = module_name

  @property
  def dof_count(self):
    return self._robot_full.dof_count

  @property
  def dof_count_ee(self):
    return self._robot_ee.dof_count

  @property
  def robot_full(self):
    return self._robot_full

  @property
  def robot_ee(self):
    return self._robot_ee

  def get_jog(self, positions, cmd_pose, cmd_vel, dt):
    robot = self._robot_ee
    dof = self._robot_ee.dof_count

    cmd_pose_xyz = np.array([cmd_pose[0, 3], cmd_pose[1, 3], cmd_pose[2, 3]])
    xyz_objective = hebi.robot_model.endeffector_position_objective(cmd_pose_xyz)
    orientation = cmd_pose[0:3, 0:3]
    theta_objective = hebi.robot_model.endeffector_so3_objective(orientation)
    new_arm_joint_angs = self._robot_ee.solve_inverse_kinematics(positions[:dof], xyz_objective, theta_objective)

    jacobian_new = self._robot_ee.get_jacobian_end_effector(new_arm_joint_angs)

    try:
      joint_velocities = np.array(np.linalg.pinv(jacobian_new)) @ np.array(cmd_vel).reshape(6,1)
    except np.linalg.LinAlgError:
      # Singular matrix
      print('No solution found. Not sending command. \n\n\n')
      new_arm_joint_angs = positions
      joint_velocities = np.zeros(dof, np.float64)

    return new_arm_joint_angs[:dof] , joint_velocities

  def get_IK_from_matrix(self, positions, cmd_pose):
    cmd_pose_xyz = np.array([cmd_pose[0, 3], cmd_pose[1, 3], cmd_pose[2, 3]])
    orientation = cmd_pose[0:3, 0:3]
    return self.get_IK(positions, cmd_pose_xyz, orientation)

  def get_IK(self, positions, ee_xyz, ee_rot):
    dof = self._robot_ee.dof_count
    xyz_objective = hebi.robot_model.endeffector_position_objective(ee_xyz)
    theta_objective = hebi.robot_model.endeffector_so3_objective(ee_rot)
    return self._robot_ee.solve_inverse_kinematics(
        positions[:dof], xyz_objective, theta_objective)

  def get_FK_ee(self, positions):
    return self._robot_ee.get_end_effector(positions[:self._robot_ee.dof_count])

  def get_grav_comp_efforts(self, feedback):
    return self._get_grav_comp_efforts(feedback.position)

  # Implementaiton follows https://github.com/HebiRobotics/hebi-python-examples/blob/3ea1988ed4b7ce1acb57f144a597a2e2478caef2/advanced/demos/random-waypoints-and-trajectories.py#L7
  def _get_grav_comp_efforts(self, positions):
    """
    Gets the torques which approximately balance out the effect
    of gravity on the arm
    """
    robot = self._robot_full
    gravityVec = np.array([0, 0, 1])
    gravity = gravityVec / np.linalg.norm(gravityVec) * 9.81

    num_frames = robot.get_frame_count('CoM')
    jacobians = robot.get_jacobians('CoM', positions)
    masses = self._masses

    comp_torque = np.zeros((robot.dof_count, 1))
    wrench_vec = np.zeros(6)
    for i in range(num_frames):
      wrench_vec[0:3] = gravity * robot.masses[i]
      comp_torque += jacobians[i].T * wrench_vec.reshape(6, 1)
    return np.squeeze(comp_torque)

def create_empty_robot_default_hrdf() -> ArmContainer:
  model_ee = hebi.robot_model.import_from_hrdf(get_hrdf_path())
  # Notice that, the custom robot model EE ends at a desired location
  # e.g. the holder of the bottom chopsticks
  # Whereas the default model leaves it at the last actuator
  model_full = hebi.robot_model.import_from_hrdf(get_hrdf_path())
  model_full.add_bracket('X5-LightBracket','Left')
  model_full.add_actuator('X5-1') # IDK why hebi-py 1.0.2 yields None here ... things are working though
  assert model_ee.dof_count == 6
  assert model_full.dof_count == 7
  return ArmContainer(None, model_full, model_ee, [])


def create_empty_robot(dof=7) -> ArmContainer:
  # Empty robot does not look up the real module
  # Used for IK and urdf and etc
  assert dof == 7
  # The End Effector Kinematic Model
  # Note that it will use all information in the FK params, but will
  # also adds a transformation to bring the EE to desired location (Refer to README)
  model_ee = create_custom_robot_model(use_last_transformation=True)

  # The Full Robot (don't use the custom transformation. Use the actual link between 6-th and 7-th joint)
  model_full = create_custom_robot_model(use_last_transformation=False)
  model_full.add_bracket('X5-LightBracket','Left')
  model_full.add_actuator('X5-1') # IDK why hebi-py 1.0.2 yields None here ... things are working though
  # TODO ensure that the two models do not interfere with each other
  assert model_ee.dof_count == 6
  assert model_full.dof_count == 7
  return ArmContainer(None, model_full, model_ee, [])

def create_robot(dof=7):
  assert dof == 7
  arm_container = create_empty_robot(dof)

  lookup = hebi.Lookup()

  # You can modify the names here to match modules found on your network
  module_families = ['Arm']
  module_names = ['J0_base', 'J1_shoulder', 'J2_elbow', 'J3_wrist1',
                  'J4_wrist2', 'J5_wrist3', 'J6_chop']

  from time import sleep
  sleep(1)

  arm = lookup.get_group_from_names(module_families, module_names)

  if arm is None:
    print('\nCould not find arm group: Did you forget to set the module family and names?')
    print('Searched for modules named:')
    print("{0} with family '{1}'".format(
      ', '.join(["'{0}'".format(entry) for entry in module_names]), module_families))

    print('Modules on the network:')
    for entry in lookup.entrylist:
      print(entry)
    else:
      print('[No Modules Found]')
    exit(1)

  arm_container.group = arm
  assert arm.size == arm_container._robot_full.dof_count
  return arm_container
