"""
Scheme is a dictionary that contains meta information of buffer data.
"vshape", "group", "init_value", "process"
"""
import torch
import numpy as np


class EpisodeBatch:
    """
    A batch that saves a group of episodic trajectories.
    Data in batch are in shape of [parallel_envs, episode_length, groups, vshape].
    """

    def __init__(self, scheme, parallel_envs, episode_length):
        self.scheme = scheme
        self.size = parallel_envs
        self.episode_length = episode_length

        self.reset()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._meta[key]
        else:
            return {k: self._meta[k][key] for k in self._meta}

    def reset(self):
        # "filled" is a reserved key of buffer
        assert "filled" not in self.scheme
        self._meta = {}
        self._meta["filled"] = torch.zeros((self.size, self.episode_length))

        for key in self.scheme:
            vshape = self.scheme[key]["vshape"]
            group = self.scheme[key]["group"]
            dtype = self.scheme[key].get("dtype", torch.float32)
            init_value = self.scheme[key].get("init_value", 0)
            process = self.scheme[key].get("process", [])
            self._meta[key] = torch.ones(
                (self.size, self.episode_length, group, *vshape), dtype=dtype) * init_value
            # process t+1: save values at timestep t+1
            if "t+1" in process:
                self._meta[key + "/t+1"] = torch.ones(
                    (self.size, self.episode_length, group, *vshape), dtype=dtype) * init_value
            # process t-1: save values at timestep t-1
            if "t-1" in process:
                self._meta[key + "/t-1"] = torch.ones(
                    (self.size, self.episode_length, group, *vshape), dtype=dtype) * init_value

    def add(self, data: dict, t): 
        assert t < self.episode_length
        self._meta["filled"][:, t] = 1
        for key in data:
            d = data[key]
            dtype = self.scheme[key].get("dtype", torch.float32)
            process = self.scheme[key].get("process", [])
            if isinstance(d, np.ndarray):
                d = torch.from_numpy(d)
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d)
            d = d.detach().cpu().clone().type(dtype)
            d = d.reshape((self.size, self.scheme[key]["group"], *self.scheme[key]["vshape"]))
            self._meta[key][:, t] = d
            if "t+1" in process and t > 0:
                self._meta[key + "/t+1"][:, t-1] = d.clone()
            if "t-1" in process and t < self.episode_length - 1:
                self._meta[key + "/t-1"][:, t+1] = d.clone()


class EpisodeBuffer(EpisodeBatch):
    """
    A replay buffer that saves episodic trajectories.
    Data in buffer are in shape of [buffer_length, episode_length, groups, vshape].
    """

    def __init__(self, scheme, buffer_size, episode_length):
        super().__init__(scheme, buffer_size, episode_length)

    def reset(self):
        super().reset()
        self.current = 0
        self.content = 0

    def insert(self, batch: EpisodeBatch):
        """
        Insert a batch into the replay buffer.
        Throw old episodes by FIFO principle.
        """
        if self.current + batch.size <= self.size:
            # no need to split
            for key in batch._meta:
                self._meta[key][self.current:self.current + batch.size] = batch._meta[key]
        else:
            # split
            split_self = self.current + batch.size - self.size
            split_batch = self.size - self.current
            for key in batch._meta:
                self._meta[key][self.current:] = batch._meta[key][:split_batch]
                self._meta[key][:split_self] = batch._meta[key][split_batch:]

        self.current = (self.current + batch.size) % self.size
        self.content = min(self.content + batch.size, self.size)

    def can_sample(self, batch_size, num_batches=1):
        return batch_size * num_batches <= self.content

    def can_sample_timestep(self, batch_size, num_batches=1):
        return batch_size * num_batches <= self.content * (self.episode_length - 1)

    def sample(self, batch_size, num_batches=1):
        """
        Sample a group of data in replay buffer to train.
        The sampled data should be in shape of [num_batches, batch_size, episode_length, group, vshape].
        """
        assert batch_size * num_batches <= self.content
        sample_num = batch_size * num_batches
        sample_index = torch.randperm(self.content)[:sample_num]

        samples = {}
        for key in self._meta:
            samples[key] = self._meta[key][sample_index]

        # cut unfilled data: [N*B, T, ...]
        max_t_filled = int(samples["filled"].sum(dim=1).max().item())
        for key in self._meta:
            a = samples[key][:, :max_t_filled]
            ori_shape = a.shape[1:]
            # [N*B, T, ...] -> [N, B, T]
            samples[key] = a.reshape(num_batches, batch_size, *ori_shape).detach().clone()
        return samples

    def sample_timestep(self, batch_size, num_batches=1):
        """
        Sample a group of data in replay buffer to train.
        The sampled data should be in shape of [num_batches, batch_size, 2, group, vshape].
        """
        assert batch_size * num_batches <= self.content * (self.episode_length - 1)
        sample_num = batch_size * num_batches
        sample_index = torch.randperm(self.content * (self.episode_length - 1))[:sample_num]

        samples = {}
        for key in self._meta:
            ori_shape = self._meta[key].shape[2:]
            a = self._meta[key][:, :-1].reshape(-1, *ori_shape)[sample_index]
            b = self._meta[key][:, 1:].reshape(-1, *ori_shape)[sample_index]
            samples[key] = torch.stack([a, b], dim=1).reshape(num_batches, batch_size, 2, *ori_shape).detach().clone()

        return samples
    
    def sample_ground_truth(self, batch_size):
        '''
        Output:
            obs: [batch_size, num_agents, obs_shape]
            belief_ground_truth: [batch_size, num_agents]
        '''
        sample_index = []
        for i in range(batch_size):
            while True:
                index = np.random.choice(self.content * self.episode_length)
                if self._meta["filled"][index // self.episode_length, index % self.episode_length]:
                    sample_index.append(index)
                    break
        return self._meta["obs"].reshape(-1, *self._meta["obs"].shape[2:])[sample_index], self._meta["ground_truth"].reshape(-1, *self._meta["ground_truth"].shape[2:])[sample_index].squeeze(-1)
