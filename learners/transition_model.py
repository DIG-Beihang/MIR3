import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.nets import MLPBase, RNNBase

class TransitionModel(nn.Module):
    """
    Transition model for environment.
    """
    def __init__(self, obs_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False):
        super(TransitionModel, self).__init__()
        
        self.n_agents = n_agents
        
        base = RNNBase if use_rnn else MLPBase
        input_shape = (obs_shape + action_shape) * self.n_agents

        self.base = base(input_shape, hidden_dim, obs_shape * self.n_agents)

    def forward(self, obs, actions, hidden_state=None):
        """
        Input:
            obs: [batch_size, num_agents, obs_shape]
            actions: [batch_size, num_agents, action_shape]
            hidden_state: [batch_size, hidden_dim]
        Output:
            out: [batch_size, num_agents, obs_shape]
            hs: [batch_size, hidden_dim]
        """
        B, G = obs.shape[:2]
        input = torch.cat([obs, actions], dim=2).reshape(B, -1)
        out, hs = self.base(input, hidden_state)

        return out.view(B, G, -1), hs
