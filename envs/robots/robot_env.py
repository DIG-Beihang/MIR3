from ..multiagentenv import MultiAgentEnv
from .envs.point_envs.rendezvous import RendezvousEnv
import numpy as np


class RobotEnv(MultiAgentEnv):
    def __init__(self, map_name, seed=0, nr_agents=5, obs_mode='sum_obs', comm_radius=40,
                 world_size=100, distance_bins=16, bearing_bins=8, torus=False, dynamics='unicycle'):
        self.map_name = map_name
        self._seed = seed
        self._env = RendezvousEnv(nr_agents=nr_agents, obs_mode=obs_mode, comm_radius=comm_radius, 
                                  world_size=world_size, distance_bins=distance_bins, bearing_bins=bearing_bins,
                                  torus=torus, dynamics=dynamics)
        self._env.seed(self._seed)
        self.state = None
        self.obs = None
        self.n_agents = nr_agents
        self.episode_limit = self._env.timestep_limit

    def step(self, actions):
        """ Returns reward, terminated, info """
        obs, state, rewards, dones, info, _ = self._env.step(actions)
        self.obs = np.array(obs)
        self.state = state
        return rewards, dones, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._env.observation_space[0].shape[0]

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self._env.share_observation_space[0].shape[0]

    def get_avail_actions(self):
        return np.ones([self.n_agents, self._env.action_space[0].shape[0]])

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones([self._env.action_space[agent_id].shape[0]])

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self._env.action_space[0].shape[0]

    def reset(self):
        """ Returns initial observations and states"""
        obs, state, _ = self._env.reset()
        self.obs = obs
        self.state = state
        return self.obs, self.state

    def render(self, mode='human'):
        self._env.render(mode=mode)

    def close(self):
        pass

    def seed(self, seed=None):
        self._env.seed(seed=seed)

    def save_replay(self, replay_dir="./"):
        return self._env.save_replay(replay_dir=replay_dir)

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info
