import numpy as np
from gym import spaces
from .robot_env import RobotEnv

class RobotWrapper:
    """
    reset() -> None
    step(action) -> reward, env_done, info

    get_obs() -> obs [N, obs_shape]
    get_state() -> state [state_shape]
    get_avail_actions -> avail_actions [N, n_actions]
    """

    def __init__(self, seed=0, obs_agent_id=True, state_agent_id=True, **args):
        self._env = RobotEnv(seed=seed, **args)
        self.obs_agent_id = obs_agent_id
        self.state_agent_id = state_agent_id
        info = self._env.get_env_info()

        if obs_agent_id:
            self.observation_spaces = [
                spaces.Box(low=0, high=1, shape=(info["obs_shape"] + self._env.n_agents,), dtype="float32")
                for _ in range(self._env.n_agents)
            ]
        else:
            self.observation_spaces = [
                spaces.Box(low=0, high=1, shape=(info["obs_shape"],), dtype="float32") for _ in range(self._env.n_agents)
            ]

        if state_agent_id:
            self.state_space = spaces.Box(low=0, high=1, shape=(info["state_shape"] + self._env.n_agents,), dtype="float32")
        else:
            self.state_space = spaces.Box(low=0, high=1, shape=(info["state_shape"],), dtype="float32")

        self.action_spaces = self._env._env.action_space
        self.n_agents = info["n_agents"]
        self.episode_limit = info["episode_limit"]

    def reset(self):
        self._env.reset()

    def get_obs(self):
        obs = self._env.get_obs()
        if self.obs_agent_id:
            agent_id_onehot = np.eye(self.n_agents)
            obs = [np.concatenate([obs[i], agent_id_onehot[i]], axis=0) for i in range(self.n_agents)]
        return np.stack(obs)

    def get_state(self):
        state = self._env.get_state().copy()
        if self.state_agent_id:
            agent_id_onehot = np.eye(self.n_agents)
            state = [np.concatenate([state[i], agent_id_onehot[i]], axis=0) for i in range(self.n_agents)]

        return np.stack(state)

    def get_avail_actions(self):
        if self._env.get_avail_actions() is not None:
            return self._env.get_avail_actions()
        else:
            return None

    def step(self, actions):
        reward, dones, info = self._env.step(actions)
        env_done = np.all(dones)
        # self._env.render(mode='animate')
        return reward, env_done, info

    def save_replay(self):
        self._env.save_replay()

    def close(self):
        self._env.close()
