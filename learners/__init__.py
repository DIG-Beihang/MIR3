from .base_learner import BaseLearner
from .maddpg_learner import MADDPGLearner
from .m3ddpg_learner import M3DDPGLearner
from .romq_learner import ROMQLearner
from .mir3_learner import MIR3Learner
from .maddpg_ca_mi_learner import MADDPGMILearner
from .ernie_learner import ERNIELearner
from .romax_learner import ROMAXLearner

REGISTRY = {
    "maddpg": MADDPGLearner,
    "m3ddpg": M3DDPGLearner,
    "mir3": MIR3Learner,
    "romq": ROMQLearner,
    "maddpg_ca_mi": MADDPGMILearner,
    "ernie": ERNIELearner,
    "romax": ROMAXLearner,
}
