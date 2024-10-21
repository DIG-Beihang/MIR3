import numpy as np
from gym import spaces
from .mujoco_multi import MujocoMulti


class MujocoWrapper:
    def __init__(self, seed=0, obs_agent_id=True, state_agent_id=True, **args) -> None:
        self._env = MujocoMulti(seed=seed, **args)
        self.obs_agent_id = obs_agent_id
        self.state_agent_id = state_agent_id
        info = self._env.get_env_info()

        if obs_agent_id:
            self.observation_spaces = [
                spaces.Box(low=-10, high=10, shape=(info["obs_shape"] + self._env.n_agents,), dtype="float32")
                for _ in range(self._env.n_agents)
            ]
        else:
            self.observation_spaces = [
                spaces.Box(low=-10, high=10, shape=(info["obs_shape"],), dtype="float32") for _ in range(self._env.n_agents)
            ]

        if state_agent_id:
            self.state_space = spaces.Box(low=-10, high=10, shape=(info["state_shape"] + self._env.n_agents,), dtype="float32")
        else:
            self.state_space = spaces.Box(low=-10, high=10, shape=(info["state_shape"],), dtype="float32")
        self.action_spaces = info["action_spaces"]
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
        state = [self._env.get_state().copy() for _ in range(self.n_agents)]
        if self.state_agent_id:
            agent_id_onehot = np.eye(self.n_agents)
            state = [np.concatenate([state[i], agent_id_onehot[i]], axis=0) for i in range(self.n_agents)]
        return np.stack(state)

    def get_avail_actions(self):
        return np.stack(self._env.get_avail_actions())

    def step(self, actions):
        reward, env_done, info = self._env.step(actions)
        reward = np.array([reward for _ in range(self.n_agents)])
        return reward, env_done, info

    def save_replay(self):
        pass

    def close(self):
        self._env.close()
