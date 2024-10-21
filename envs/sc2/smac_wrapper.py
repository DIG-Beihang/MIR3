"""
The step() of StarCraft2Env has possible returns below:

0, True, {} full restart
reward, False, {"battle_won": False, "dead_enemies": xxx, "dead_allies": xxx}
reward, True, {"battle_won": False, "dead_enemies": xxx, "dead_allies": xxx, "episode_limit": True}
reward, True, {"battle_won": xxx, "dead_enemies": xxx, "dead_allies": xxx}
"""

import numpy as np
from gym import spaces
from .starcraft2 import StarCraft2Env


class SMACWrapper:
    """
    reset() -> None
    step(action) -> reward, env_done, info

    get_obs() -> obs [N, obs_shape]
    get_state() -> state [state_shape]
    get_avail_actions -> avail_actions [N, n_actions]
    """

    def __init__(self, seed=0, obs_agent_id=True, state_agent_id=True, death_masking=True, **args):
        self._env = StarCraft2Env(seed=seed, **args)
        self.obs_agent_id = obs_agent_id
        self.state_agent_id = state_agent_id
        self.death_masking = death_masking
        info = self._env.get_env_info()

        if obs_agent_id:
            self.observation_spaces = [
                spaces.Box(low=-1, high=1, shape=(info["obs_shape"] + self._env.n_agents,), dtype="float32")
                for _ in range(self._env.n_agents)
            ]
        else:
            self.observation_spaces = [
                spaces.Box(low=-1, high=1, shape=(info["obs_shape"],), dtype="float32") for _ in range(self._env.n_agents)
            ]

        if state_agent_id:
            self.state_space = spaces.Box(low=-1, high=1, shape=(info["state_shape"] + self._env.n_agents,), dtype="float32")
        else:
            self.state_space = spaces.Box(low=-1, high=1, shape=(info["state_shape"],), dtype="float32")

        self.action_spaces = [spaces.Discrete(info["n_actions"]) for _ in range(self._env.n_agents)]
        self.n_agents = self._env.n_agents
        self.episode_limit = self._env.episode_limit

    def reset(self):
        self._env.reset()

    def get_obs(self):
        obs = self._env.get_obs()
        if self.obs_agent_id:
            agent_id_onehot = np.eye(self.n_agents)
            obs = [np.concatenate([obs[i], agent_id_onehot[i]], axis=0) for i in range(self.n_agents)]
        return np.stack(obs)

    def get_state(self):
        ori_state = self._env.get_state()
        state = []

        for i in range(self.n_agents):
            agent_info = self._env.get_unit_by_id(i)
            if self.death_masking and agent_info.health == 0:
                state.append(np.zeros_like(ori_state))
            else:
                state.append(ori_state.copy())

        if self.state_agent_id:
            agent_id_onehot = np.eye(self.n_agents)
            state = [np.concatenate([state[i], agent_id_onehot[i]], axis=0) for i in range(self.n_agents)]

        return np.stack(state)

    def get_avail_actions(self):
        return np.stack(self._env.get_avail_actions())

    def step(self, actions):
        reward, env_done, info = self._env.step(actions)
        reward = np.array([reward for _ in range(self.n_agents)])

        alive_agents = []
        for i in range(self.n_agents):
            agent_info = self._env.get_unit_by_id(i)
            if agent_info.health == 0:
                alive_agents.append(False)
            else:
                alive_agents.append(True)
        info["alive_agents"] = np.array(alive_agents)

        return reward, env_done, info

    def save_replay(self):
        self._env.save_replay()

    def close(self):
        self._env.close()
