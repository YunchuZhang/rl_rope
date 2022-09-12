from __future__ import print_function

import numpy as np
from scipy.spatial.transform import Rotation as scipyR

from .arm_container import create_empty_robot
from .smoother import Smoother
from .utils import simple_iir_lowpass

class HebiPIDController():
  # Pass a target, pass it through PID
  def __init__(self, config, step_size=0.01):
    self.config = {k:float(v) for k,v in config.items()}
    self.config['ki'] *= step_size
    self.config['kd'] /= step_size
    self.config['d_on_error'] = (self.config['d_on_error'] == '1')
    self.feed_forward = self.config['feed_forward']
    #print(self.config)
    # TODO support extra parameters:
    # kp ki kd feed_forward dead_zone i_clamp punch min_target max_target target_lowpass min_output max_output output_lowpass d_on_error
    # we care about kp ki kd i_clamp min_target max_target target_lowpass min_ouptut max_output output_lowpass d_on_error
    # we don't care about dead_zone punch
    self.reset()

  def reset(self):
    self.errSum = 0
    self.lastDOn = 0
    self.last_target = None
    self.last_output = None

  def update(self, observed, target, feedforward=0):
    if target is None and (not feedforward):
      return 0
    target = target or 0
    if feedforward:
      target += feedforward * self.feed_forward

    target = np.clip(target, self.config['min_target'], self.config['max_target'])
    target = simple_iir_lowpass(target, self.last_target, self.config['target_lowpass'])
    self.last_target = target
    error = target - observed
    self.errSum += error
    if self.config['d_on_error']:
      dErr = (error - self.lastDOn)
      self.lastDOn = error
    else:
      dErr = self.lastDOn - observed # note the flip, derivative kick
      self.lastDOn = observed
    i_output = np.clip(self.config['ki'] * self.errSum, -self.config['i_clamp'], self.config['i_clamp'])
    output = self.config['kp'] * error + i_output + self.config['kd'] * dErr
    output = np.clip(output, self.config['min_output'], self.config['max_output'])
    output = simple_iir_lowpass(output, self.last_output, self.config['output_lowpass'])
    self.last_output = output
    return output

class HebiJointController():
  # Given target position / velocity / effort target for a joint, transform to PWM
  def __init__(self, config):
    self.config = config
    if int(config['control_strategy']) == 3:
      self.update = self._ctrl_strategy3
    elif int(config['control_strategy']) == 4:
      self.update = self._ctrl_strategy4
    else:
      print('Do not support the given control_strategy, int(control_strategy)={}'.format(int(config['control_strategy'])))
      raise NotImplementedError
    self.ctrl = \
      {k : HebiPIDController(self.config[v]) for k,v in {'p':'position','v':'velocity','e':'effort'}.items()}

  def reset(self):
    [v.reset() for v in self.ctrl.values()]

  def _ctrl_strategy3(self, p_, v_, e_, p, v, e): # observed + target -> output
    _p = self.ctrl['p'].update(p_, p)
    _v = self.ctrl['v'].update(v_, v)
    _e = self.ctrl['e'].update(e_, e)
    return _p + _e + _v

  def _ctrl_strategy4(self, p_, v_, e_, p, v, e): # observed + target -> output
    intermediate_effort = self.ctrl['p'].update(p_, p)
    _e = self.ctrl['e'].update(e_, intermediate_effort + e)
    _v = self.ctrl['v'].update(v_, v)
    return _e + _v

class HebiJointPositionController(HebiPIDController):
  def __init__(self, config):
    super().__init__(config['position'])

  def update(self, p_, v_, e_, p, v, e):
    return super().update(p_, p)

def create_joint_controller(xml_config, onlyPosition=False):
  # Factory func, parse gains xml to produce JointController
  def iter_xml(element):
    if len(element) == 0:
      return element.text.split()
    return {child.tag: iter_xml(child) for child in element}
  def sub_dict(item, index):
    if isinstance(item, list):
      return item[index]
    return {k: sub_dict(v, index) for k,v in item.items()}
  import xml.etree.ElementTree as ET
  #print('xml_config', xml_config)
  tree = ET.parse(xml_config)
  ctrl_dict = iter_xml(tree.getroot())
  ctrl = []
  for i in range(7):
    sub_ctrl_dict = sub_dict(ctrl_dict, i)
    ctrl.append(
      HebiJointPositionController(sub_ctrl_dict) if onlyPosition
      else HebiJointController(sub_ctrl_dict))
  return ctrl

class TychoController:
  # Given a target EE pose, transform to PWM output
  def __init__(self, gains_config_xml, onlyPosition=False):
    self.arm = create_empty_robot()
    self.load_gains(gains_config_xml, onlyPosition)
    self.reset()
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.10f},'.format}, linewidth=1000)

  def load_gains(self, gains_config_xml, onlyPosition=False):
    self.joint_controllers = create_joint_controller(gains_config_xml, onlyPosition)

  def reset(self):
    self.smoother = Smoother(7, 30)
    [ctrl.reset() for ctrl in self.joint_controllers]

  def act(self, target_pos, target_vel, pos_, vel_, effort_):
    if target_pos is not None:
      self.smoother.append(target_pos)
      pos = self.smoother.get()
    else:
      pos = [None] * 7
    effort = self.arm._get_grav_comp_efforts(pos_)
    vel = target_vel if target_vel is not None else ([None] * 7)
    # import pdb;pdb.set_trace()
    pwm_outputs = [ctrl.update(p_, v_, e_, p,v,e) for ctrl,p_, v_, e_, p,v,e in zip(self.joint_controllers, pos_, vel_, effort_,  pos, vel, effort)]
    #print('pos_fdbk',pos_[-1], 'pos_cmd', pos[-1])
    #print('-> pwm',pwm_outputs[-1])
    # import pdb;pdb.set_trace()
    return pwm_outputs

  # Pass observed and target command in, first observed then target
  def gen_pwm(self, pos_, vel_, effort_, pos, vel, effort):
    pwm = [ctrl.update(p_, v_, e_, p,v,e) for ctrl,p_, v_, e_, p,v,e in zip(self.joint_controllers, pos_, vel_, effort_,  pos, vel, effort)]
    #if np.isnan(pwm[0]):
    #  import IPython; IPython.embed()
    return pwm

if __name__ == '__main__':
  # Create a dummy controller
  # Deprecated hrdf = '/Users/cakey/dev/hebi_teleop/gains/chopstick7D.hrdf'
  gains = '/Users/cakey/dev/hebi_teleop/gains/chopstick-gains-7D.xml'
  create_joint_controller(gains)
  dummy_controller = TychoController(gains)

  # Test the pwm output against a recording
  gains = '/Users/cakey/Downloads/with_efforts_without_LP.xml'
  recording_fn = '/Users/cakey/Downloads/with_efforts_without_LP.csv'
  test_controller = TychoController(gains)
  import tqdm, pandas as pd
  import pdb; pdb.set_trace()
  recording = pd.read_csv(recording_fn)
  acc_error = np.zeros((recording.shape[0], 7))
  for index, row in recording.iterrows():
    if index == recording.shape[0]:
      break
    pos_, vel_, effort_, pos, vel, effort, pwm_ = [
      np.fromstring(r, dtype=np.float, sep=' ')
      for r in [row['positions'], row['velocities'], row['efforts'],
      row['positionCommands'], row['velocityCommands'], row['effortCommands'],
      row['pwmCommands']]]
    #print([len(a) for a in [pos_, vel_, effort_, pos, vel, effort, pwm_]])
    pwm = np.array(test_controller.gen_pwm(pos_, vel_, effort_, pos, vel, effort))
    #print(pwm)
    #print(pwm_)
    acc_error[index] = pwm - pwm_
  #print(acc_error)

  print(np.linalg.norm(acc_error, axis=1))

  import IPython
  IPython.embed()
