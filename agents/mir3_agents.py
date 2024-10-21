import os
import copy
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .base_agents import BaseAgents
from .nets import MLPBase, RNNBase, MINE, CLUBContinuous, CLUBCategorical

class MIR3Actor(nn.Module):
    """
    Actor network (mu) for MIR3 algorithms.
    The output of actor network is a deterministic action.
    """
    
    def __init__(self, obs_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False, share_params=True):
        super(MIR3Actor, self).__init__()

        self.n_agents = n_agents
        self.share_params = share_params

        base = RNNBase if use_rnn else MLPBase
        if share_params:
            self.base = base(obs_shape, hidden_dim, action_shape)
        else:
            self.base = nn.ModuleList(
                [base(obs_shape, hidden_dim, action_shape) for _ in range(n_agents)])

    def forward(self, obs, hidden_state, avail_actions=None):
        """
        Input:
            obs: [batch_size, num_agents, obs_shape]
            hidden_state: [batch_size, num_agents, hidden_dim]
            avail_actions: [batch_size, num_agents, action_shape]
        Output:
            out: [batch_size, num_agents, action_shape]
            hs: [batch_size, num_agents, hidden_dim]
        """
        if self.share_params:
            out, hs = self.base(obs, hidden_state)
        else:
            assert obs.shape[1] == self.n_agents
            out = []
            hs = []
            for i in range(self.n_agents):
                if hidden_state is not None:
                    t1, t2 = self.base[i](obs[:, i], hidden_state[:, i])
                else:
                    t1, t2 = self.base[i](obs[:, i], None)
                out.append(t1)
                hs.append(t2)
            out = torch.stack(out, dim=1)
            if hs[0] is not None:
                hs = torch.stack(hs, dim=1)
            else:
                hs = None

        if avail_actions is not None:
            out[avail_actions == 0] = -1e10

        return out, hs


class MIR3Critic(nn.Module):
    """
    Critic network (Q) for MIR3 Algorithms.
    The input of critic network are global states and joint actions;
    the output of critic network are Q-values.
    """

    def __init__(self, state_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False, share_params=True):
        super(MIR3Critic, self).__init__()

        self.n_agents = n_agents
        self.share_params = share_params

        base = RNNBase if use_rnn else MLPBase
        if share_params:
            self.base = base(state_shape + n_agents * action_shape, hidden_dim, 1)
        else:
            self.base = nn.ModuleList(
                [base(state_shape + n_agents * action_shape, hidden_dim, 1) for _ in range(n_agents)])

    def forward(self, state, actions, hidden_state):
        """
        Input:
            state: [batch_size, num_agents, obs_shape]
            actions: [batch_size, num_agents, action_shape]
            hidden_state: [batch_size, num_agents, hidden_dim]
        Output:
            out: [batch_size, num_agents, 1]
            hs: [batch_size, num_agents, hidden_dim]
        """
        
        B, N, A = actions.shape
        assert N == self.n_agents

        joint_actions = actions.unsqueeze(1).repeat(1, N, 1, 1).reshape(B, N, N*A)
        inputs = torch.cat([state, joint_actions], dim=2)
        if self.share_params:
            out, hs = self.base(inputs, hidden_state)
        else:
            out = []
            hs = []
            for i in range(self.n_agents):
                if hidden_state is not None:
                    t1, t2 = self.base[i](inputs[:, i], hidden_state[:, i])
                else:
                    t1, t2 = self.base[i](inputs[:, i], None)
                out.append(t1)
                hs.append(t2)
            out = torch.stack(out, dim=1)
            if hs[0] is not None:
                hs = torch.stack(hs, dim=1)
            else:
                hs = None

        return out, hs


class MIR3Agents(BaseAgents):
    def __init__(self, obs_shape, state_shape, action_shape, action_type, n_agents,
                 hidden_dim=64, use_rnn=False, share_params=True, device="cuda:0"):
        assert action_type in ["discrete", "box"]
        self.hidden_dim = hidden_dim
        self.action_type = action_type
        self.n_agents = n_agents
        self.device = device

        self.actor = MIR3Actor(obs_shape, action_shape, n_agents, hidden_dim, use_rnn, share_params)
        self.critic = MIR3Critic(state_shape, action_shape, n_agents, hidden_dim, use_rnn=False, share_params=share_params)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        self.mine = MINE(state_shape, n_agents * action_shape, hidden_dim)
        if self.action_type == "discrete":
            self.club = CLUBCategorical(state_shape, n_agents * action_shape, hidden_dim)
        if self.action_type == "box":
            self.club = CLUBContinuous(state_shape, n_agents * action_shape, hidden_dim)
        
        self.to(self.device)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.n_agents, self.hidden_dim)).to(self.device)

    def perform(self, obs, hidden_state=None, available_actions=None):
        obs = self.check(obs)
        if hidden_state is not None:
            hidden_state = self.check(hidden_state)
        if available_actions is not None:
            available_actions = self.check(available_actions)
        actions, hs = self.actor(obs, hidden_state, available_actions)
        if self.action_type == "discrete":
            actions = actions.argmax(dim=-1)
        if self.action_type == "box":
            actions = torch.tanh(actions)
        return actions, hs

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        self.mine.to(device)
        self.club.to(device)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.mine.state_dict(), os.path.join(path, "mine.pth"))
        torch.save(self.club.state_dict(), os.path.join(path, "club.pth"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(
            path, "actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(
            path, "critic.pth"), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.mine.load_state_dict(torch.load(os.path.join(path, "mine.pth"), map_location=self.device))
        self.club.load_state_dict(torch.load(os.path.join(path, "club.pth"), map_location=self.device))

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()
        self.mine.train()
        self.club.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        self.mine.eval()
        self.club.eval()
        
    def update_targets_hard(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update_targets_soft(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
