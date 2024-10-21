from absl import flags

from .sc2.smac_wrapper import SMACWrapper
from .mujoco.mujoco_wrapper import MujocoWrapper
from .robots.robot_wrapper import RobotWrapper

FLAGS = flags.FLAGS
FLAGS(["main.py"])

REGISTRY = {
    "sc2": SMACWrapper,
    "mujoco": MujocoWrapper,
    "robot": RobotWrapper
}
