"""
Vectorized environments.
In parallel envs, terminated envs will be automated reset.
"""

import numpy as np
from multiprocessing import Pipe, Process
from functools import partial


def env_worker(remote, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, env_done, info = env.step(actions)
            remote.send((reward, env_done, info))
        elif cmd == "reset":
            env.reset()
        elif cmd == "get_state":
            remote.send(env.get_state())
        elif cmd == "get_obs":
            remote.send(env.get_obs())
        elif cmd == "get_avail_actions":
            remote.send(env.get_avail_actions())
        elif cmd == "save_replay":
            env.save_replay(**data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError


class BaseEnv:
    def reset(self):
        """
        reset the environment.
        no args and no returns.
        """
        raise NotImplementedError

    def step(self, actions):
        """
        perform a one-time step.
        args:
            actions: np.ndarray with shape [n_envs, n_agents, action_shape]
        returns:
            reward: np.ndarray with shape [n_envs, n_agents,]
            done: np.ndarray with shape [n_envs, n_agents,]
            info: list with shape [n_envs,]
        """
        raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class DummyEnv(BaseEnv):
    def __init__(self, env_fn, num_envs, episode_limit, seed=0, env_args={}):
        self.env_fn = env_fn
        self.num_envs = num_envs
        self._envs = [env_fn(seed=seed + i * 1000, **env_args) for i in range(num_envs)]
        self._env_t = [-1 for _ in range(num_envs)]

        self.episode_limit = episode_limit

    def reset(self):
        for i in range(self.num_envs):
            if self._env_t[i] != 0:
                self._envs[i].reset()
                self._env_t[i] = 0

    def get_obs(self):
        obs_batch = []
        for i in range(self.num_envs):
            obs = self._envs[i].get_obs()
            obs_batch.append(obs)

        return np.stack(obs_batch)

    def get_state(self):
        state_batch = []
        for i in range(self.num_envs):
            state = self._envs[i].get_state()
            state_batch.append(state)

        return np.stack(state_batch)

    def get_avail_actions(self):
        avail_actions_batch = []
        for i in range(self.num_envs):
            avail_actions = self._envs[i].get_avail_actions()
            avail_actions_batch.append(avail_actions)

        if avail_actions_batch[0] is None:
            return None
        return np.stack(avail_actions_batch)

    def step(self, actions):
        reward_batch = []
        done_batch = []
        info_batch = []
        for i in range(self.num_envs):
            assert self._env_t[i] >= 0
            reward, done, info = self._envs[i].step(actions[i])
            self._env_t[i] += 1

            if done or self._env_t[i] >= self.episode_limit:
                self._envs[i].reset()
                self._env_t[i] = 0

            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)

        return (
            np.stack(reward_batch),
            np.stack(done_batch),
            info_batch,
        )

    def close(self):
        for env in self._envs:
            env.close()

    def save_replay(self, **args):
        self._envs[0].save_replay(**args)


class SubProcVecEnv(BaseEnv):
    def __init__(self, env_fn, num_envs, episode_limit, seed=0, env_args={}):
        self.env_fn = env_fn
        self.num_envs = num_envs

        self._env_t = [-1 for _ in range(self.num_envs)]
        self.episode_limit = episode_limit

        self._envs, self.worker_conns = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(
                target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_args, seed=i * 1000 + seed))),
            )
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

    def reset(self):
        for i in range(self.num_envs):
            if self._env_t[i] != 0:
                self._envs[i].send(("reset", None))
                self._env_t[i] = 0

    def get_state(self):
        state_batch = []
        for i in range(self.num_envs):
            self._envs[i].send(("get_state", None))

        for i in range(self.num_envs):
            state = self._envs[i].recv()
            state_batch.append(state)

        return np.stack(state_batch)

    def get_obs(self):
        obs_batch = []
        for i in range(self.num_envs):
            self._envs[i].send(("get_obs", None))

        for i in range(self.num_envs):
            obs = self._envs[i].recv()
            obs_batch.append(obs)

        return np.stack(obs_batch)

    def get_avail_actions(self):
        avail_actions_batch = []
        for i in range(self.num_envs):
            self._envs[i].send(("get_avail_actions", None))

        for i in range(self.num_envs):
            avail_actions = self._envs[i].recv()
            avail_actions_batch.append(avail_actions)

        if avail_actions_batch[0] is None:
            return None
        return np.stack(avail_actions_batch)

    def step(self, actions):
        reward_batch = []
        done_batch = []
        info_batch = []
        for i in range(self.num_envs):
            assert self._env_t[i] >= 0
            self._envs[i].send(("step", actions[i]))

        for i in range(self.num_envs):
            reward, env_done, info = self._envs[i].recv()
            self._env_t[i] += 1

            if env_done or self._env_t[i] >= self.episode_limit:
                self._envs[i].send(("reset", None))
                self._env_t[i] = 0

            reward_batch.append(reward)
            done_batch.append(env_done)
            info_batch.append(info)

        return (
            np.stack(reward_batch),
            np.stack(done_batch),
            info_batch,
        )

    def close(self):
        for env in self._envs:
            env.send(("close", None))

    def save_replay(self, **args):
        self._envs[0].send(("save_replay", args))
