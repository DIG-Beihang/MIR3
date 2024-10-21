import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agents import BaseAgents
from .nets import MLPBase, MLPBase_relax, RNNBase, RNNBase_relax

def onehot_from_logits(logits):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


class ROMAXActor(nn.Module):
    """
    Actor network (\mu) for ROMAX algorithms.
    The output of actor network is a deterministic action.
    """

    def __init__(self, obs_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False, share_params=True):
        super(ROMAXActor, self).__init__()

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
            #get the action-vector 
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

class MinQNetwork(nn.Module):
    """
    MinQ network (\mu) for adversarial training.
    The output of minQ network is the Q-value of every action.
    """

    def __init__(self, obs_shape, action_shape, hidden_dim=64, use_rnn=False):
        super(MinQNetwork, self).__init__()

        base = RNNBase if use_rnn else MLPBase
        self.base = base(obs_shape, hidden_dim, action_shape)

    def forward(self, obs, hidden_state, avail_actions=None):
        """
        Input:
            input: [batch_size, num_agents, input_shape]
            hidden_state: [batch_size, num_agents, hidden_dim]
            avail_actions: [batch_size, num_agents, action_shape]
        Output:
            out: [batch_size, num_agents, action_shape]
            hs: [batch_size, num_agents, hidden_dim]
        """
        out, hs = self.base(obs, hidden_state)

        if avail_actions is not None:
            out[avail_actions == 0] = 1e10

        return out, hs

class ROMAXCritic(nn.Module):
    """
    Critic network (Q) for ROMAX Algorithms.
    The input of critic network are global states and joint actions;
    the output of critic network are Q-values.
    """

    def __init__(self, state_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False, share_params=True):
        super(ROMAXCritic, self).__init__()

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

class ROMAXCritic_relax(nn.Module):
    """
    Critic network (Q) for ROMAX Algorithms.
    The input of critic network are global states and joint actions;
    the output of critic network are Q-values.
    """

    def __init__(self, state_shape, action_shape, n_agents, hidden_dim=64, use_rnn=False, share_params=True):
        super(ROMAXCritic_relax, self).__init__()

        self.n_agents = n_agents
        self.share_params = share_params

        base = RNNBase_relax if use_rnn else MLPBase_relax
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


class ROMAXAgents(BaseAgents):
    def __init__(self, obs_shape, state_shape, action_shape, action_type, n_agents,
                 hidden_dim=64, use_rnn=False, share_params=True, device="cuda:0"):
        assert action_type in ["discrete", "box"]
        self.obs_shape = obs_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.action_type = action_type
        self.n_agents = n_agents
        self.device = device

        self.actor = ROMAXActor(obs_shape, action_shape,
                               n_agents, hidden_dim, use_rnn, share_params)
        self.critic = ROMAXCritic(state_shape, action_shape, 
                                 n_agents, hidden_dim, use_rnn=False, share_params=share_params)
        self.critic_relax = ROMAXCritic_relax(state_shape, action_shape, 
                                 n_agents, hidden_dim, use_rnn=False, share_params=share_params)
        self.minq_net = MinQNetwork(obs_shape, action_shape, hidden_dim, use_rnn=False)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic_relax = copy.deepcopy(self.critic_relax)
        self.target_minq_net = copy.deepcopy(self.minq_net)

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
            # actions = torch.clamp(actions, -1, 1)
        return actions, hs

    def perform_logits(self, obs, hidden_state=None, available_actions=None):
        obs = self.check(obs)
        if hidden_state is not None:
            hidden_state = self.check(hidden_state)
        if available_actions is not None:
            available_actions = self.check(available_actions)
        actions, hs = self.actor(obs, hidden_state, available_actions)
        action_logits = actions.clone()
        if self.action_type == "discrete":
            actions = actions.argmax(dim=-1)
        if self.action_type == "box":
            actions = torch.tanh(actions)
            # actions = torch.clamp(actions, -1, 1)
        return actions, hs, action_logits
    
    def perform_action_representation(self, obs, hidden_state=None):
        obs = self.check(obs)
        if hidden_state is not None:
            hidden_state = self.check(hidden_state)
        return self.actor.base.representation(obs, hidden_state)

    def attack_obs(self, obs_ori, actor_hs, avail_actions, state, adversary, iter, epsilon, alpha, target_actor=None, target_hs=None):
        # obs_ori: [N, G, obs_shape]
        obs_ori = self.check(obs_ori)
        state = self.check(state)
        if actor_hs is not None:
            actor_hs = self.check(actor_hs)
        if avail_actions is not None:
            avail_actions = self.check(avail_actions)
        obs = obs_ori.detach().clone()
        for i in range(iter):
            obs.requires_grad_(True)
            actions, hs = self.actor(obs, actor_hs, avail_actions)
            if target_actor is not None:
                if target_hs is not None:
                    target_actions, _ = target_actor(obs[:, adversary], target_hs, avail_actions[:, adversary])
                elif actor_hs is not None:
                    target_actions, _ = target_actor(obs[:, adversary], actor_hs[:, adversary], avail_actions[:, adversary])
                else:
                    target_actions, _ = target_actor(obs[:, adversary], None, avail_actions[:, adversary])
                loss = torch.norm(target_actions - actions[:, adversary], p=1)
            else:
                ori_actions, ori_hs = self.actor(obs_ori, actor_hs, avail_actions)
                loss = -torch.norm(actions[:, adversary] - ori_actions[:, adversary])
            # if self.action_type == "discrete":
            #     actions_onehot = F.gumbel_softmax(actions, hard=True)
            # elif self.action_type == "box":
            #     actions_onehot = torch.tanh(actions)
            # q_value, _ = self.critic(state, actions_onehot, None)
            # q_value.sum().backward()
            loss.backward()
            obs_grad = obs.grad.sign()[:, adversary]
            obs = obs.detach()
            obs[:, adversary] -= alpha* obs_grad
            obs = (torch.clamp(obs - obs_ori, -epsilon, epsilon) + obs_ori).detach()
        return obs

    def attack_action(self, actions_ori, actions_logits_ori, avail_actions, state, adversary, iter, epsilon, alpha):
        # actions_ori (box): [N, G, n_actions]
        # actions_ori (discrete): [N, G]
        # actions_logits_ori: [N, G, n_actions]
        if self.action_type == "discrete":
            actions, _ = self.attack_minq(actions_ori, avail_actions, state, adversary)
            return actions
        actions = self.check(actions_logits_ori.detach().clone())
        state = self.check(state)
        for _ in range(iter):
            actions.requires_grad_(True)
            if self.action_type == "box":
                actions_onehot = torch.tanh(actions)
            else:
                raise NotImplementedError
            q_value, _ = self.critic(state, actions_onehot, None)
            q_value.sum().backward()
            action_grad = actions.grad.sign()[:, adversary]
            actions = actions.detach()
            actions[:, adversary] -= alpha * action_grad
            actions = (torch.clamp(actions - actions_ori, -epsilon, epsilon) + actions_ori).detach()
        return torch.tanh(actions)

    def attack_minq(self, actions_ori, avail_actions, state, adversary):
        # actions_ori (discrete): [N, G]
        # avail_actions: [N, G, actions_shape]
        actions_ori = self.check(actions_ori)
        avail_actions = self.check(avail_actions)
        state = self.check(state)
        if self.action_type != "discrete":
            raise NotImplementedError
        actions_shape = avail_actions.shape[-1]
        actions_onehot = F.one_hot(actions_ori.to(torch.int64), num_classes=actions_shape)
        q_values = []
        for i in range(actions_shape):
            actions = actions_onehot.detach().clone()
            actions[:, adversary] = 0
            actions[:, adversary, i] = 1
            q_value, _ = self.critic(state, actions, None)
            q_values.append(q_value)
        q_values = torch.cat(q_values, dim=2)
        q_values_clone = q_values.detach().clone().cpu().numpy()
        q_values = q_values.sum(dim=1) # [N, actions_shape]
        q_values[avail_actions[:, adversary] == 0] = 1e10
        actions = q_values.argmin(dim=1) # [N,]

        actions_ori = actions_ori.detach().clone()
        actions_ori[:, adversary] = actions

        return actions_ori, q_values.min(dim=-1).values
    
    def attack_m3ddpg(self, action_logits, avail_actions, state, adv_epsilon):
        action_logits = self.check(action_logits)
        avail_actions = self.check(avail_actions)
        state = self.check(state)
        if self.action_type == 'discrete':
            action_onehot = onehot_from_logits(action_logits)
        elif self.action_type == 'box':
            action_onehot = torch.tanh(action_logits)
        else:           
            raise NotImplementedError
        actions_copy = action_onehot.detach().clone()
        actions_copy.requires_grad_(True)
        q, _ = self.critic(state, actions_copy, None)
        q.sum().backward()
        action_noise = actions_copy.grad.data.clone().detach()
        actions_copy = actions_copy - action_noise * adv_epsilon
        if self.action_type == 'discrete':
            actions_copy = actions_copy - actions_copy.min(dim=-1, keepdim=True).values
            actions_copy[avail_actions == 0] = 0.0
            actions_prob = actions_copy / actions_copy.sum(dim=-1, keepdim=True)
            actions = Categorical(actions_prob).sample()
        else:
            actions = torch.clamp(actions_copy, -1, 1).detach()
        return actions

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.critic_relax.to(device)
        self.minq_net.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        self.target_critic_relax.to(device)
        self.target_minq_net.to(device)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.critic_relax.state_dict(), os.path.join(path, "critic_relax.pth"))
        torch.save(self.minq_net.state_dict(), os.path.join(path, "minq.pth"))

    def load(self, path):
        # try:
        self.actor.load_state_dict(torch.load(os.path.join(
            path, "actor.pth"), map_location=self.device))
        # except:
        #     print("Load actor checkpoints from", path, "error!")
        try:
            self.critic.load_state_dict(torch.load(os.path.join(
                path, "critic.pth"), map_location=self.device))
            self.critic_relax.load_state_dict(torch.load(os.path.join(
                path, "critic_relax.pth"), map_location=self.device))
        except:
            print("Load critic checkpoints from", path, "error!")
        try:
            self.minq_net.load_state_dict(torch.load(os.path.join(
                path, "minq.pth"), map_location=self.device))
        except:
            print("Load minq network checkpoints from", path, "error!")
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_relax.load_state_dict(self.critic_relax.state_dict())
        self.target_minq_net.load_state_dict(self.minq_net.state_dict())

    def prep_training(self):
        self.actor.train()
        self.critic.train()
        self.minq_net.train()
        self.target_actor.train()
        self.target_critic.train()
        self.target_critic_relax.train()
        self.target_minq_net.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        self.minq_net.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        self.target_critic_relax.eval()
        self.target_minq_net.eval()

    def update_targets_hard(self):
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_relax.load_state_dict(self.critic_relax.state_dict())
        self.target_minq_net.load_state_dict(self.minq_net.state_dict())

    def update_targets_soft(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_critic_relax.parameters(), self.critic_relax.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_minq_net.parameters(), self.minq_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
