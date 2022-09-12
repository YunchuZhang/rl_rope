from distutils.core import setup

setup(
    name='tycho_env',
    version='1.0',
    packages=['tycho_env','tycho_env.utils'],
    license='TODO',
    long_description=open('README.md').read(),
)

# ROS installation 
## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD
#from distutils.core import setup
#from catkin_pkg.python_setup import generate_distutils_setup
# fetch values from package.xml
#setup_args = generate_distutils_setup(
#    packages=['tycho_env'],
#    package_dir={'': 'src'},
#)
#setup(**setup_args)
