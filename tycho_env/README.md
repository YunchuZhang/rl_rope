# tycho_env

- OpenAI gym env for hebi, wrapper around mujoco model
- Utility functions for hebi

## Install
Support python 3 only.
You might need to `pip install hebi-py=1.0.2` dependency.
`pip install -e .` install this library as editable!
For python2, switch to py2 branch which installs only the utility functions but not the mujoco env.

## Updating Kinematic Model Following Calibration

In `utils/utils.py`.

- Replace your new R matrix for the kinematic model. It should contains 7 numbers (though the last one is not used).

- Replace your new FK parameter matrix. It should contains DH parameters for 6 links and a transformation (qx qy qz qw x y z) to the tip of the bottom chopsticks (which are tracked).

## Parameters

If you change how the chopsticks are mounted, you need to change all the transformation function from EE to tip and update URDF of robot model.

1. `arm_container.py` computes FK given joint positions. It defines where FK point is at. Starting from 2020.07.21 (last updated 2021 June) the FK point is at the center of holder of the bottom chopsticks, where the holder is screwed and touches the plate. Note this point is not exactly on the top plate of the actuator. Also note that most likely the calibration script will transform the EE to the tip of the chopsticks. We add a fixed transformation to the robot ee model to bring it to the desired EE point.

2. `utils/util.py` has definition to transform EE pose to chopsticks "middle" tip and vice versa. The chopsticks "middle" tip is at the middle point of two chopsticks tips. To compute it, we recover the rotation center (pm) and the tip of bottom chopsticks (p1). Rotate the pm_to_p1 vector by half of the opening angle then scale by cos of the opening angle, we obtain pm_to_tip.


## Custom Env

There is a separate branch, `custom_env` that stores the latest stable version of the environment.

To use it, clone the repo to a folder named `tycho_env_custom`; checkout the `custom-env` branch; `pip install -e .`; in your application script `import tycho_env_custom` just like how you would import normal `tycho_env`.

You can have both `tycho_env` and `tycho_env_custom` in the same workspace and both installed by pip, as long as they are in separate folders. But you should only use one of them in the application script. When you import `tycho_env_custom` it will shadow `tycho_env` to be invisible (so it can use its own version of `tycho_env`).

To update the content in custom env, `git checkout custom_env; git merge master;`. 