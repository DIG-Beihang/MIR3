from .base_agents import BaseAgents
from .ddpg_agents import DDPGAgents
from .mir3_agents import MIR3Agents
from .romax_agents import ROMAXAgents

REGISTRY = {
    "ddpg": DDPGAgents, 
    "mir3": MIR3Agents,
    "romax": ROMAXAgents,
}
