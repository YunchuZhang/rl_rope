import math, numpy as np, os, sys
from time import time
from scipy.spatial.transform import Rotation as scipyR
import collections

#######################################################################
# Physical Parameters
#######################################################################

# Chopsticks angle updated on 2022 June 2
# Ensure the fix chopsticks tip is exactly 10 cm from the side of the holding bracket
CHOPSTICK_OPEN = -0.1755           # unit rad
CHOPSTICK_PARALLEL = -0.375747        # unit rad
CHOPSTICK_CLOSE = -0.597          # unit rad
tip_hsv = [[73,139,20],[109,255,255]]
ball_hsv = [[0,172,0],[255,255,255]]
R_OPTITRACK2BASE =np.array([ # Specified as qx qy qz qw x y z
  # Updated on 2022 May.30
  -0.00162249, 0.00215902, -0.00000002, 0.99999635, -0.93919181, 0.16715677, -0.00317523])

FK_PARAMS = np.array( # Specified as alpha, a, d
# Updated 2022 June 1
  [-0.00002708896322369181, 0.00000000000000000000, 0.10340924641018893471, -0.02618534102554722492,
 1.57094467418716376983, 0.00000000000000000000, 0.08246846422923261033, -0.01316785268315446254,
 3.14160825775615615285, 0.32392523580014997986, 0.04532389380435437876, 0.00015046944537916779,
 3.14007445293408693487, 0.32786656091612614849, 0.07116201007080100172, 0.00000627481008851037,
 1.57079545315256097204, 0.00000000000000000000, 0.11431079260606799575, -0.00092215476185221685,
 1.57079351667003042081, 0.00000000000000000000, 0.11446253877193594828, 0.00001239363281179647,
 -0.70699999999999996181, 0.00000000000000000000, 0.00000000000000000000, 0.70699999999999996181,
 0.11744807583186629707, 0.07875778669732630410, 0.02516253872496395511])

MOVING_POSITION = [-1.69926117, 1.91009097, 2.09709026, -0.09968156, 1.5817661, 0.07704814, -0.375747]
# MOVING_POSITION = [-1.616640849068654, 2.255957540501213, 2.034268882672058, -0.22171666090491185, 1.5778532138690338, 0.02503266214570154, -0.4181700646877289]



OFFSET_JOINTS = np.zeros(7)
OFFSET_JOINTS[:6] = FK_PARAMS[:-7].reshape(6,4)[:,-1].reshape(6)

SMOOTHER_WINDOW_SIZE = 30

DH_params = np.zeros((7,3))
DH_params[:6,:] = FK_PARAMS[:-7].reshape((6,-1))[:6, :3] # each link is represented by the first 3 params
DH_params[-1, :] = [-1.5707983, 0.00, 0.11373140497245554092/2]

#######################################################################
# Arm Utilities
#######################################################################

def get_res_estimator_path():
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'assets')
    model_path = os.path.join(dir_path, 'res-estimator.pt')
    return model_path

def get_hrdf_path():
  # The HRDF is likely to only contain the first 6 actuators
  # Irregardless of the filename
  dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'assets')
  hrdf_file = os.path.join(dir_path, 'chopstick7D.hrdf')
  return hrdf_file

def get_gains_path(suffix='', DOF=7):
  dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'assets')
  xml_file = os.path.join(dir_path, 'chopstick-gains-{}D{}.xml'.format(DOF, suffix))
  return xml_file

def load_gain(group, gain_xml_fn):
  import hebi
  group_cmd = hebi.GroupCommand(group.size)
  group_cmd.read_gains(gain_xml_fn)
  group.send_command_with_acknowledgement(group_cmd)
  print('loaded gain from ', gain_xml_fn)

def save_gain(group, gain_xml_fn):
  group_info = group.request_info()
  if group_info is not None:
    group_info.write_gains(gain_xml_fn)
    print('saved gain to {}'.format(gain_xml_fn))

#######################################################################
# Angle Utilities
#######################################################################

def axis_angle_from_rotation_matrix(matrix):
  axis = np.zeros(3)
  axis[0] = matrix[2,1] - matrix[1,2]
  axis[1] = matrix[0,2] - matrix[2,0]
  axis[2] = matrix[1,0] - matrix[0,1]
  t = matrix[0,0] + matrix[1,1] + matrix[2,2]
  r = np.linalg.norm(axis)
  theta = math.atan2(r, t-1)
  axis = axis / r
  return axis, theta

def angle_difference_between_rotation_matrix(matA, matB):
  diffMat = matA * matB.transpose()
  trMat = diffMat[0,0] + diffMat[1,1] + diffMat[2,2]
  return np.arccos((trMat - 1) / 2)

def quat_to_euler_angles(qx, qy, qz, qw):
    quat = (qx, qy, qz, qw)
    params = quat + (0,) * 3
    return euler_angles_from_rotation_matrix(get_transformation_matrix_from_quat(params)[:3,:3])

def euler_angles_to_quat(angles):
    """
    Angles represents the euler angles, represented as intrinsic rotations in the order ZYX.
    Dimensions are either (n_rot, 3) or (3,). In the latter case, a single quat will be returned,
    in the former case, multiple quats will be returned.
    """
    return scipyR.from_euler("ZYX", angles).as_quat()

def get_transformation_matrix_from_quat(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_matrix()
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

R_optitrack2base = get_transformation_matrix_from_quat(R_OPTITRACK2BASE)

def R_axis_angle(matrix, axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle (radian).
    @type angle:    float
    """

    # Trig factors.
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = 1 - ca
    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca

def euler_angles_from_rotation_matrix(R):
  '''
  From a paper by Gregory G. Slabaugh (undated),
  "Computing Euler angles from a rotation matrix
  '''
  phi = 0.0
  if np.isclose(R[2,0], -1.0):
      theta = math.pi/2.0
      psi = math.atan2(R[0,1],R[0,2])
  elif np.isclose(R[2,0], 1.0):
      theta = -math.pi/2.0
      psi = math.atan2(-R[0,1],-R[0,2])
  else:
      theta = -math.asin(R[2,0])
      cos_theta = math.cos(theta)
      psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
      phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
  return psi, theta, phi

def rotation_matrix_from_euler_angles(roll, pitch, yaw):
  theta = np.array([roll, pitch, yaw])

  R_x = np.array([[1,         0,                  0                   ],
                  [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                  [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                  ])
  R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                  [0,                     1,      0                   ],
                  [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                  ])
  R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                  [math.sin(theta[2]),    math.cos(theta[2]),     0],
                  [0,                     0,                      1]
                  ])
  R = np.dot(R_z, np.dot( R_y, R_x ))
  return R

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_inv(quat):
    w, x, y, z = quat
    return np.array([w, -x, -y, -z])

def rotation_matrix_between_two_vectors(i_v, unit=None):
    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)
    # Get axis
    uvw = np.cross(i_v, unit)
    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)
    #normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw
    # Compute rotation matrix - re-expressed to show structure
    return (
        rcos * np.eye(3) +
        rsin * np.array([
            [ 0, -w,  v],
            [ w,  0, -u],
            [-v,  u,  0]
        ]) +
        (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    )

#######################################################################
# Construct Target Utilities
#######################################################################

def simple_iir_lowpass(x, y, alpha):
  return x if (y is None or alpha == 1.0) else y + alpha * (x-y)

def construct_choppose(arm, current_position, target_at_ee=True):
  """transfer current joint position(7 joints angle) to be EndEffector pose (xyz + quaternion + chop_open_angle)
  Args:
    current_position: current joint position(6 joints angle + last joint as chopstick open angle).
    target_at_ee: whether to use EndEffector pose as state representation.
  Returns:
    choppose: state representation for imitation learning, formed as EndEffector pose (xyz + quaternion + chop_open_angle)
  """
  if not target_at_ee:
      import warnings
      warnings.warn("EE position now refers to the chopstick tip. "
                    "Usage of the target_at_ee flag is deprecated, and does not do anything.", DeprecationWarning)
  current_transformation = np.array(arm.get_FK_ee(current_position))
  current_open = current_position[-1]
  choppose = np.zeros(8)
  choppose[3:7] = scipyR.from_matrix(current_transformation[0:3,0:3]).as_quat()
  choppose[7] = current_open
  choppose[0:3] = current_transformation[0:3,3]
  return choppose

#######################################################################
# Construct Command Utilities
#######################################################################

# construct command for an ee pose
# specified either as 8D vector containing xyz + quat + open
# or as 3 by 4 matrix transformation and an opening angle
def construct_command(arm, current_joint_position,
                      target_vector=None,
                      target_transformation=None,
                      target_open=None):
  next_angles = list(current_joint_position)
  if target_vector is not None:
    target_transformation = np.zeros((3,4))
    target_transformation[0:3, 3] = target_vector[0:3]
    target_transformation[0:3, 0:3] = scipyR.from_quat(target_vector[3:7]).as_matrix()
    target_open = target_vector[-1]
  next_angles[:-1] = arm.get_IK_from_matrix(current_joint_position, target_transformation)
  if target_open:
    next_angles[-1] = np.clip(target_open, CHOPSTICK_CLOSE, CHOPSTICK_OPEN)
  return next_angles

#######################################################################
# ROS
#######################################################################

def imgMsgToImg(camera_msg, opt): # TODO replace cv2
  import cv2
  img_data = np.fromstring(camera_msg.data, np.uint8)
  img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
  if opt[0]:
    img = img[opt[0]:opt[0]+opt[1],opt[2]:opt[2]+opt[3],:]
  if opt[6]:
    img = cv2.resize(img, (opt[4], opt[5]), interpolation = opt[6])
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # The returned image is of shape HxWxC. Channel order RGB. Values 0~255.
  return img

################################################################################
# Chopsticks mounting
################################################################################

def transform_tip_to_chop_middle(tip1, quat, chop_angle):
  """
  tip1 is xyz position of static chopstick, in robot space.
      Can be either (3,) single coord or (n,3) matrix of coords
  quat is orientation of EE in robot space.
      Can be either (4,) single quat or (n,4) matrix of quats
  chop_angle is the angle of the chopsticks
      Can be either single float or (n,) array of floats

  Returns the point between the chopsticks. If n chopstick tip poses
  were passed, returns (n,3) matrix of points. If a single tip pose
  was passed (without the batch dimension) then a single (3,) point.
  """
  if len(tip1.shape) == 1:
    tip1 = np.expand_dims(tip1, axis=0)
  if len(quat.shape) == 1:
    quat = np.expand_dims(quat, axis=0)
  if isinstance(chop_angle, (int, float)) or len(chop_angle.shape) == 0:
    chop_angle = np.expand_dims(float(chop_angle), axis=0)
  TIP_TO_PIVOT = np.array([-FK_PARAMS[-3], FK_PARAMS[-1], 0]).reshape(-1,1) # in ee local space
  rot_mat = scipyR.from_quat(quat).as_matrix() # CoB from local space to global
  chop_angle = np.where(chop_angle > CHOPSTICK_CLOSE, chop_angle - CHOPSTICK_CLOSE, 0)
  pivot = tip1 + np.squeeze(rot_mat @ TIP_TO_PIVOT)
  z_axis = rot_mat[..., -1] # remember +z = down
  tan_angles = np.tan(chop_angle / 4.).reshape(-1,1)
  mrp_rots = z_axis * np.broadcast_to(tan_angles, z_axis.shape)
  tip2_rot_mat = scipyR.from_mrp(mrp_rots).as_matrix() @ rot_mat
  tip2 = pivot + np.squeeze(tip2_rot_mat @ -TIP_TO_PIVOT)
  return np.squeeze((tip1 + tip2) / 2.0)

def transform_chop_middle_to_tip(chop_middle, quat, chop_angle):
  """
  chop_middle is xyz position of point between chopsticks, in robot space.
      Can be either (3,) single coord or (n,3) matrix of coords
  quat is orientation of EE in robot space.
      Can be either (4,) single quat or (n,4) matrix of quats
  chop_angle is the angle of the chopsticks
      Can be either single float or (n,) array of floats

  Returns the position of the static chopstick. If n poses
  were passed, returns (n,3) matrix of points. If a single pose
  was passed (without the batch dimension) then a single (3,) point.
  """
  if len(chop_middle.shape) == 1:
    chop_middle = np.expand_dims(chop_middle, axis=0)
  if len(quat.shape) == 1:
    quat = np.expand_dims(quat, axis=0)
  if isinstance(chop_angle, (int, float)) or len(chop_angle.shape) == 0:
    chop_angle = np.expand_dims(float(chop_angle), axis=0)
  TIP_TO_PIVOT = np.array([-FK_PARAMS[-3], FK_PARAMS[-1], 0]).reshape(-1,1) # in ee local space
  rot_mat = scipyR.from_quat(quat).as_matrix()
  chop_angle = np.where(chop_angle > CHOPSTICK_CLOSE, chop_angle - CHOPSTICK_CLOSE, 0)
  pivot_to_tip1 = np.squeeze(rot_mat @ -TIP_TO_PIVOT) # in global space
  z_axis = rot_mat[..., -1]
  tan_angles = np.tan(chop_angle/4).reshape(-1,1)
  mrp_rots = z_axis * np.broadcast_to(tan_angles, z_axis.shape)
  tip2_rot_mat = scipyR.from_mrp(mrp_rots).as_matrix() @ rot_mat
  tip2_to_pivot = np.squeeze(tip2_rot_mat @ TIP_TO_PIVOT)
  mid_to_tip1 = (pivot_to_tip1 + tip2_to_pivot) / 2.0
  return np.squeeze(chop_middle + mid_to_tip1)

def transform_from_middle_of_tip_to_ee_center(position, orientation, chopopen, debug=False): # TODO rename
  if debug:
    x,y,z = position # DEBUG
    return x,y,z
  rotation_matrix = scipyR.from_quat(orientation).as_matrix()
  x_axis, y_axis, z_axis = rotation_matrix.T

  # !!!!!!!!!!!
  # Accept p1 location
  # !!!!!!!!!!!
  p1 = np.array(position)
  po = p1 - x_axis * 0.1135 - y_axis * 0.007 - z_axis * 0.0073 # Origin of bottom chopsticks on robot, also the screw hole or ee
  x,y,z = po
  return x,y,z

  pm_to_p1 = x_axis * 0.1285 - y_axis * 0.03 # THIS SHOULD FOLLOW transform_ee_to_tip
  q1 = np.array([0] + list(pm_to_p1))
  rotate = chopopen - CHOPSTICK_CLOSE
  rotate = rotate if rotate >= 0 else 0.0
  rotate = rotate / 2
  rx, ry, rz = z_axis
  q2 = np.array([np.cos(rotate/2), rx * np.sin(rotate/2), ry * np.sin(rotate/2), rz * np.sin(rotate/2)])
  qprod = q_mult(q_mult(q2, q1), q_inv(q2))
  pm_to_pt = np.array(qprod[1:4]) * np.cos(rotate)
  pm = np.array(position) - pm_to_pt
  p1 = pm + pm_to_p1
  po = p1 - x_axis * 0.1135 - y_axis * 0.007 - z_axis * 0.0073 # Origin of bottom chopsticks on robot, also the screw hole or ee
  x,y,z = po
  return x,y,z

def transform_choppose_to_tip(choppose):
  _p = choppose.pose.position
  _q = choppose.pose.orientation
  _o = choppose.open
  position = np.array([_p.x, _p.y, _p.z])
  rotation_matrix = scipyR.from_quat([_q.x, _q.y, _q.z, _q.w]).as_matrix()
  return transform_ee_to_tip(position, rotation_matrix, _o)

def get_ee_from_tracker(raw_points):
  # TODO need to update to the tip position
  # Given the three points on the EE tracker, return EE pose
  raw_points = np.hstack((raw_points, np.ones((3,1))))
  three_points = [R_optitrack2base.dot(_p)[0:3] for _p in raw_points]
  point1, point2, point3 = three_points # from left to right
  x_axis = np.array(point2 - point3)
  x_axis /= np.linalg.norm(x_axis)
  vector1 = np.array(point1 - point2)
  vector1 /= np.linalg.norm(vector1)
  z_axis = np.cross(x_axis, vector1)
  z_axis /= np.linalg.norm(z_axis)
  y_axis = np.cross(z_axis, x_axis)
  y_axis /= np.linalg.norm(y_axis)
  rotation_matrix = np.array([x_axis,y_axis,z_axis]).T
  quat = scipyR.from_matrix(rotation_matrix).as_quat()
  # First move to the center of the hollow hole
  opoint = point2 + 0 * x_axis + 0.045 * y_axis + (0.006 + 0.022 + 0.005) * z_axis # the original EE point by hebi FK (if no additional rigid body), the last Lightweight Bracket hollow hole center, 6mm = ball_center_to_bottom, 0.022 = holder_height, 0.005 = height of bracket
  eepoint = opoint + 0.015 * x_axis + (-0.03) * y_axis + (0.03 - 0.0017) * z_axis # bottom of holder, apply the same transformation as the added rigid body in arm_container.py

  return eepoint, quat

##########################
# For calibrating FK
##########################

def get_DH_transformation(alpha,a,_theta,d,theta_offset=0):
  theta = _theta + theta_offset
  # Given the DH link construct the transformation matrix
  rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha)],
                 [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha)]])

  trans = np.array([a,-d*np.sin(alpha),np.cos(alpha)*d]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  m = np.vstack((np.hstack((rot, trans)),last_row))
  return m

def get_transformation_matrix(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_matrix()
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def calculate_FK_transformation(FKparams, joint_position):
  # Given a list of FKparams, shape N by 3, return transformation
  ee = np.eye(4)
  for (alpha, a, d), theta in zip(FKparams, joint_position):
    ee = ee.dot(get_DH_transformation(alpha, a, theta, d))
  return ee

def get_fk_tips(list_jp, _fk_params=FK_PARAMS):
  dh_params = np.copy(_fk_params[:-7].reshape((6,-1))[:6, :3]) # each link is represented by the first 3 params
  last_transformation = get_transformation_matrix(_fk_params[-7:])
  list_fk_tips = []
  for jp in list_jp:
    # print(f"this is joint position = {jp}")
    ee = calculate_FK_transformation(dh_params, jp)
    ee = ee.dot(last_transformation)
    list_fk_tips.append(ee[0:3, 3])
  return np.array(list_fk_tips).reshape(-1,3)

################################################################################
# Misc Utils
################################################################################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Python program to print
# colored text and background
class colors:
    reset='\033[0m'
    bold='\033[01m'
    disable='\033[02m'
    underline='\033[04m'
    reverse='\033[07m'
    strikethrough='\033[09m'
    invisible='\033[08m'
    class fg:
        black='\033[30m'
        red='\033[31m'
        green='\033[32m'
        orange='\033[33m'
        blue='\033[34m'
        purple='\033[35m'
        cyan='\033[36m'
        lightgrey='\033[37m'
        darkgrey='\033[90m'
        lightred='\033[91m'
        lightgreen='\033[92m'
        yellow='\033[93m'
        lightblue='\033[94m'
        pink='\033[95m'
        lightcyan='\033[96m'
    class bg:
        black='\033[40m'
        red='\033[41m'
        green='\033[42m'
        orange='\033[43m'
        blue='\033[44m'
        purple='\033[45m'
        cyan='\033[46m'
        lightgrey='\033[47m'

def print_and_cr(msg): sys.stdout.write(msg + '\r\n')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Namespace(collections.MutableMapping):
    """Utility class to convert a (nested) dictionary into a (nested) namespace.
    >>> x = Namespace({'foo': 1, 'bar': 2})
    >>> x.foo
    1
    >>> x.bar
    2
    >>> x.baz
    Traceback (most recent call last):
        ...
    KeyError: 'baz'
    >>> x
    {'foo': 1, 'bar': 2}
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': 1, 'bar': 2}
    >>> x = Namespace({'foo': {'a': 1, 'b': 2}, 'bar': 3})
    >>> x.foo.a
    1
    >>> x.foo.b
    2
    >>> x.bar
    3
    >>> (lambda **kwargs: print(kwargs))(**x)
    {'foo': {'a': 1, 'b': 2}, 'bar': 3}
    >>> (lambda **kwargs: print(kwargs))(**x.foo)
    {'a': 1, 'b': 2}
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, k):
        if not k.startswith('_'):
            if k not in self._data:
                return Namespace({})
            v = self._data[k]
            if isinstance(v, dict):
                v = Namespace(v)
            return v

        if k not in self.__dict__:
            raise AttributeError("'Namespace' object has no attribute '{}'".format(k))

        return self.__dict__[k]

    def __setattr__(self, key, value):
        if not key.startswith('_'):
            self._data[key] = value
        else:
            self.__dict__[key] = value

    def __repr__(self):
        return repr(self._data)

def parse_overrides(config, overrides):
    """
    Overrides the values specified in the config with values.
    config: (Nested) dictionary of parameters
    overrides: Parameters to override and new values to assign. Nested
        parameters are specified via dot notation.
    >>> parse_overrides({}, [])
    {}
    >>> parse_overrides({}, ['a'])
    Traceback (most recent call last):
      ...
    ValueError: invalid override list
    >>> parse_overrides({'a': 1}, [])
    {'a': 1}
    >>> parse_overrides({'a': 1}, ['a', 2])
    {'a': 2}
    >>> parse_overrides({'a': 1}, ['b', 2])
    Traceback (most recent call last):
      ...
    KeyError: 'b'
    >>> parse_overrides({'a': 0.5}, ['a', 'test'])
    Traceback (most recent call last):
      ...
    ValueError: could not convert string to float: 'test'
    >>> parse_overrides(
    ...    {'a': {'b': 1, 'c': 1.2}, 'd': 3},
    ...    ['d', 1, 'a.b', 3, 'a.c', 5])
    {'a': {'b': 3, 'c': 5.0}, 'd': 1}
    """
    if len(overrides) % 2 != 0:
        # print('Overrides must be of the form [PARAM VALUE]*:', ' '.join(overrides))
        raise ValueError('invalid override list')

    for param, value in zip(overrides[::2], overrides[1::2]):
        keys = param.split('.')
        params = config
        for k in keys[:-1]:
            if k not in params:
                raise KeyError(param)
            params = params[k]
        if keys[-1] not in params:
            raise KeyError(param)

        current_type = type(params[keys[-1]])
        value = current_type(value)  # cast to existing type
        params[keys[-1]] = value

    return config

if __name__ == "__main__":

    # verify that transforming to tip and back to ee is valid
    position = np.array([0.01, 0.03, 0.05])
    rotation_matrix = scipyR.from_quat([1.2, 0, 0.45, -0.2]).as_matrix()
    chopopen = CHOPSTICK_PARALLEL + 0.1
    middles = transform_tip_to_chop_middle(position, np.array([1.2, 0, 0.45, -0.2]), chopopen)
    import pdb;pdb.set_trace()
    tip_position = transform_ee_to_tip(position, rotation_matrix, chopopen)
    quat = scipyR.from_matrix(rotation_matrix).as_quat()
    recovered_ee_position = transform_from_middle_of_tip_to_ee_center(tip_position, quat, chopopen)
    print('position:', position)
    print('recovered position: ', recovered_ee_position)
    print('check that delta is small:' , np.linalg.norm(position-recovered_ee_position))
