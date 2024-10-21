import torch
import numpy as np

class BaseAgents:
    def check(self, input):
        output = torch.from_numpy(input) if type(input) == np.ndarray else input
        return output.to(dtype=torch.float32, device=self.device)

    def init_hidden(self, batch_size):
        """Initialize RNN hidden states."""
        raise NotImplementedError

    def perform(self, obs, hidden_state, available_actions=None):
        """Perform an action during evaluations."""
        raise NotImplementedError

    def to(self, device):
        """Move networks onto a device."""
        raise NotImplementedError

    def load(self, path):
        """Load checkpoints of networks."""
        raise NotImplementedError

    def save(self, path):
        """Save checkpoints of networks"""
        raise NotImplementedError

    def prep_training(self):
        """Set to train mode."""
        raise NotImplementedError

    def prep_rollout(self):
        """Set to eval mode."""
        raise NotImplementedError