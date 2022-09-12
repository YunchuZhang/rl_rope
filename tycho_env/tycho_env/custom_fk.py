# Create a custom kinematic model for our 7-joint robot.
# Given FK parameters
# (6 links * 3 parameters each  + 1 transformation with 7 parameters)
# generate a hebi robot arm.

import hebi, numpy as np
from scipy.spatial.transform import Rotation as scipyR

from .utils import FK_PARAMS

def get_transformation_matrix_from_quat(params):
  # Given 7 params containing quat (x,y,z,w) and shift (x,y,z) return transformation matrix
  qx,qy,qz,qw,x,y,z = params
  rot = scipyR.from_quat((qx, qy, qz, qw)).as_matrix()
  trans = np.array([x,y,z]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  return np.vstack((np.hstack((rot, trans)),last_row))

def get_DH_transformation(alpha,a,d,_theta=0,theta_offset=0):
  # Given the DH link construct the transformation matrix
  theta = _theta + theta_offset
  rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha)],
                 [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha)]])
  trans = np.array([a,-d*np.sin(alpha),np.cos(alpha)*d]).reshape(3,1)
  last_row = np.array([[0, 0, 0, 1]])
  m = np.vstack((np.hstack((rot, trans)),last_row))
  return m

def shift_z(_z):
  return np.matrix([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,_z],
                   [0,0,0,1]])

def get_origin_matrix(_x,_y,_z):
  return np.matrix([[1,0,0,_x],
                   [0,1,0,_y],
                   [0,0,1,_z],
                   [0,0,0,1]])

def add_body(my_model,
             origin,          #=np.identity(4),
             inertial,        #=np.zeros(6),
             mass,            #=0.1,
             transformation,  #=np.identity(4)
             ):
  my_model.add_rigid_body(origin, inertial, mass, transformation, False)

def create_custom_robot_model(
  use_last_transformation=True):
  # Inertia copied from URDF
  # CoM copied from URDF
  # Mass copied from URDF
  newmodel = hebi.robot_model.RobotModel()
  # Actuators
  FK = FK_PARAMS[:-7].reshape(6,-1)[:6, :3]
  # To use the FK parameters
  # We need to save some space for the actual actuators
  # We do this by offseting the d values
  DH_trans1=get_DH_transformation(FK[0,0],FK[0,1],FK[0,2]-0.0451-0.055,0)
  add_body(newmodel,
    origin=np.eye(4),                         # CoM
    inertial=np.array([0.0,0.0,0,0,0,0]),     # Inertia
    mass=0,                                   # Mass
    transformation=DH_trans1)
  newmodel.add_actuator('X8-9')
  DH_trans2=get_DH_transformation(FK[1,0],FK[1,1],FK[1,2]-0.0451,0)
  res=np.dot(shift_z(0.055),DH_trans2)
  add_body(newmodel,
    origin=get_origin_matrix(0, -0.01875, .0275),
    inertial=np.array([0.0003096,0.0003096,0.0003096,0,0,0]),
    mass=0.215+0.042,
    transformation=res)
  newmodel.add_actuator('X8-16')
  DH_trans3=get_DH_transformation(FK[2,0],FK[2,1],FK[2,2]-0.0451,0)
  add_body(newmodel,
    origin=get_origin_matrix(0.163666517345, 0, 0.02),
    inertial=np.array([0,0.00359775425495,0.00359775425495,0,0,0]),
    mass=0.402933213876+0.042,
    transformation=DH_trans3)
  newmodel.add_actuator('X8-9')
  DH_trans4=get_DH_transformation(FK[3,0],FK[3,1],FK[3,2]-0.03105-0.04,0)
  add_body(newmodel,
    origin=get_origin_matrix(0.162326322007, 0, 0.02),
    inertial=np.array([0,0.00352965749567,0.00352965749567,0,0,0]),
    mass=0.401861057606+0.042,
    transformation=DH_trans4)
  newmodel.add_actuator('X5-9')
  DH_trans5=get_DH_transformation(FK[4,0],FK[4,1],FK[4,2]-0.03105-0.04,0)
  DH_trans5=np.dot(shift_z(0.04),DH_trans5)
  add_body(newmodel,
    origin=get_origin_matrix(0, -.0215, .02),
    inertial=np.array([0.000144,0.000144,0.000144,0,0,0]),
    mass=0.1+0.042,
    transformation=DH_trans5)
  newmodel.add_actuator('X5-1')
  DH_trans6=get_DH_transformation(FK[5,0],FK[5,1],FK[5,2]-0.04-0.03105,0)
  DH_trans6=np.dot(shift_z(0.04),DH_trans6)
  add_body(newmodel,
    origin=get_origin_matrix(0, -.0215, .02),
    inertial=np.array([0.000144,0.000144,0.000144,0,0,0]),
    mass=0.1+0.042,
    transformation=DH_trans6)
  newmodel.add_actuator('X5-1')
  if use_last_transformation:
    # from the second last joint to the fixed chopstick tip and then go to the ee
    last_trans_params=FK_PARAMS[-7:]
    last_transformation=get_transformation_matrix_from_quat(last_trans_params)
    trans_res=np.dot(shift_z(0.04),last_transformation)
    add_body(newmodel,
      origin=get_origin_matrix(0, -.0215, .02),
      inertial=np.array([0.000144,0.000144,0.000144,0,0,0]),
      mass=0.1+0.315+0.003*4+0.003*5+0.042,
      transformation=trans_res)
  return newmodel

if __name__ == '__main__':
  model = create_custom_robot_model()
