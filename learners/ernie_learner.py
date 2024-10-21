import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_learner import BaseLearner


def onehot_from_logits(logits):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


class ERNIELearner(BaseLearner):
    def __init__(self, agents, noise_scale=0.1, start_steps=10000, gamma=0.99, use_adam=True, actor_lr=5e-4,
                 critic_lr=5e-4, optim_alpha=0.99, optim_eps=1e-5, max_grad_norm=10, action_reg=0.001,
                 target_update_hard=False, target_update_interval=0.01, actor_regularizer_eps=0.1, actor_regularizer_random_init=0.01, actor_regularizer_coeff=0.5,
                 critic_regularizer_eps=0.1, critic_regularizer_prob=0.1, critic_regularizer_coeff=0.5):
        self.agents = agents
        self.device = self.agents.device
        self.action_type = self.agents.action_type
        self.noise_scale = noise_scale
        self.start_steps = start_steps
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.action_reg = action_reg
        self.target_update_hard = target_update_hard
        self.target_update_interval = target_update_interval
        self.last_update_target = 0
        self.actor_regularizer_eps = actor_regularizer_eps
        self.actor_regularizer_coeff = actor_regularizer_coeff
        self.actor_regularizer_random_init = actor_regularizer_random_init
        self.critic_regularizer_eps = critic_regularizer_eps
        self.critic_regularizer_coeff = critic_regularizer_coeff
        self.critic_regularizer_prob = critic_regularizer_prob

        self.actor_params = list(self.agents.actor.parameters())
        self.critic_params = list(self.agents.critic.parameters())
        if use_adam:
            self.actor_optimizer = torch.optim.Adam(
                self.agents.actor.parameters(), lr=actor_lr, eps=optim_eps)
            self.critic_optimizer = torch.optim.Adam(
                self.agents.critic.parameters(), lr=critic_lr, eps=optim_eps)
        else:
            self.actor_optimizer = torch.optim.RMSprop(
                self.agents.actor.parameters(), lr=actor_lr, alpha=optim_alpha, eps=optim_eps)
            self.critic_optimizer = torch.optim.RMSprop(
                self.agents.critic.parameters(), lr=critic_lr, alpha=optim_alpha, eps=optim_eps)

    def collect(self, batch, t, global_t, actor_hs=None):
        obs = batch["obs"][:, t].to(self.device)
        avail_actions = batch["avail_actions"][:, t].to(self.device)
        if global_t > self.start_steps:
            actions, hs = self.agents.actor(obs, actor_hs, avail_actions)
            if self.action_type == "discrete":
                actions_onehot = F.gumbel_softmax(actions, hard=True)
                actions = actions_onehot.argmax(dim=-1)
            elif self.action_type == "box":
                actions = torch.tanh(actions)
                actions += torch.randn_like(actions) * self.noise_scale
                actions = torch.clamp(actions, -1, 1)
                actions_onehot = actions.clone()
        else: 
            hs = actor_hs
            if self.action_type == "discrete":
                actions = torch.rand_like(avail_actions)
                actions[avail_actions == 0] = -1e10
                actions_onehot = F.gumbel_softmax(actions, hard=True)
                actions = actions_onehot.argmax(dim=-1)
            elif self.action_type == "box":
                actions = torch.rand_like(avail_actions) * 2 - 1
                actions_onehot = actions.clone()

        batch.add({"actions": actions, "actions_onehot": actions_onehot}, t)

        return actions, hs
    
    def get_action_noise(self, critic, state, actor_out):
        """
        Generate one-step noise for actions.
        critic: nn.Module, critic or target_critic
        state: [B, T, G, state_shape]
        actor_out: [B, T, G, action_shape]
        """
        actions_copy = actor_out.detach().clone()
        actions_copy.requires_grad_(True)
        view_shape = state.shape[0] * state.shape[1]
        q, _ = critic(state.view(view_shape, self.agents.n_agents, -1), 
                             actions_copy.view(view_shape, self.agents.n_agents, -1), None)
        q.sum().backward()
        action_noise = actions_copy.grad.data.clone().detach()
        return action_noise
    
    def get_actor_regularizer_reward(self, obs_ori, avail_actions):
        B, T, N, V = obs_ori.shape
        obs = obs_ori.detach().clone()
        obs = torch.randn(obs.shape).to(self.device) * self.actor_regularizer_random_init + obs
        obs.requires_grad_(True)
        
        out_list = []
        hs = self.agents.init_hidden(B)
        for t in range(T):
            out, hs = self.agents.actor(obs[:, t], hs, avail_actions[:, t])
            out_list.append(out)
        out = torch.stack(out_list, dim=1)
        
        out_ori_list = []
        hs = self.agents.init_hidden(B)
        for t in range(T):
            out_ori, hs = self.agents.actor(obs_ori[:, t], hs, avail_actions[:, t])
            out_ori_list.append(out_ori)
        out_ori = torch.stack(out_ori_list, dim=1)

        out_diff = torch.norm(out - out_ori, p=1)
        out_diff.backward()
        obs_grad = obs.grad.sign()
        obs = obs.detach() + self.actor_regularizer_eps * obs_grad

        out_list = []
        hs = self.agents.init_hidden(B)
        for t in range(T):
            out, hs = self.agents.actor(obs[:, t], hs, avail_actions[:, t])
            out_list.append(out)
        out = torch.stack(out_list, dim=1)        
        return -torch.norm(out - out_ori, p=1, dim=-1, keepdim=True)
    
    def get_critic_regularizer_reward(self, actions_ori, avail_actions, states):
        B, T, N, V = actions_ori.shape
        actions_ori = actions_ori.reshape(B * T, N, -1)
        avail_actions = avail_actions.reshape(B * T, N, -1)
        states = states.reshape(B * T, N, -1)
        if self.agents.action_type == "discrete":
            actions_adv = actions_ori.detach().clone()
            actions_ori_single = torch.argmax(actions_ori, dim=-1)
            for adversary in range(N):
                actions_single, _ = self.agents.attack_minq(actions_ori_single, avail_actions, states, adversary)
                actions_single = F.one_hot(actions_single.to(torch.int64), num_classes=V)
                actions_adv[:, adversary] = actions_single[:, adversary]
            choice_adv = torch.rand([B * T, N, 1]).to(self.device) < self.critic_regularizer_prob
            actions_per = actions_adv * choice_adv + actions_ori * ~choice_adv
        else:
            actions_per = actions_ori.detach().clone().requires_grad_(True)
            q_value, _ = self.agents.critic(states, actions_per, None)
            q_value.sum().backward()
            actions_grad = actions_per.grad.sign()
            actions_per = actions_per.detach() - self.critic_regularizer_eps * actions_grad
        q_ori, _ = self.agents.critic(states, actions_ori, None)
        q_per, _ = self.agents.critic(states, actions_per, None)
        out_diff = torch.norm(q_ori - q_per, p=1, dim=-1, keepdim=True)
        return -out_diff.reshape(B, T, N, -1)       

    def learn(self, samples, episodes):
        train_info = {
            "critic_loss": 0,
            "critic_grad_norm": 0,
            "actor_loss": 0,
            "actor_grad_norm": 0,
            "q_taken_mean": 0,
            "target_mean": 0,
            "actions_mean": 0,
            "actions_norm_mean": 0
        }
        # shape: [N, B, T, G, V]
        N, B, T = samples["filled"].shape
        for n in range(N):
            obs = samples["obs"][n].to(self.device)
            state = samples["state"][n].to(self.device)
            actions = samples["actions_onehot"][n].to(self.device)
            avail_actions = samples["avail_actions"][n].to(self.device)
            rewards = samples["rewards"][n].to(self.device)
            masks = samples["masks"][n].to(self.device).expand_as(rewards)

            rewards_actor_regularizer = self.get_actor_regularizer_reward(obs, avail_actions)
            rewards_critic_regularizer = self.get_critic_regularizer_reward(actions, avail_actions, state)
            rewards = rewards + self.actor_regularizer_coeff * rewards_actor_regularizer + self.critic_regularizer_coeff * rewards_critic_regularizer

            if masks[:, :-1].sum() == 0:
                continue

            # compute target actions in t+1
            target_actor_out = []
            target_hs = self.agents.init_hidden(B)
            for t in range(T):
                target_out, target_hs = self.agents.target_actor(
                    obs[:, t], target_hs, avail_actions[:, t])
                target_actor_out.append(target_out)
            # shape: [B, T, G, action_shape]
            target_actor_out = torch.stack(target_actor_out, dim=1).detach()
            if self.action_type == "discrete":
                target_actor_out = onehot_from_logits(target_actor_out)
            if self.action_type == "box":
                target_actor_out = torch.tanh(target_actor_out)
                # target_actor_out = torch.clamp(target_actor_out, -1, 1)

            # compute target q-values in t+1
            # target_critic_out = []
            # target_hs = self.agents.init_hidden(B)
            # for t in range(T):
            #     target_out, target_hs = self.agents.target_critic(
            #         state[:, t], target_actor_out[:, t], target_hs)
            #     target_critic_out.append(target_out)
            # # shape: [B, T, G, 1]
            # target_critic_out = torch.stack(target_critic_out, dim=1)
            target_critic_out, _ = self.agents.target_critic(state.view(
                B*T, self.agents.n_agents, -1), target_actor_out.view(B*T, self.agents.n_agents, -1), None)
            target_critic_out = target_critic_out.view(
                B, T, self.agents.n_agents, -1)

            # compute q-values in t
            # critic_out = []
            # hs = self.agents.init_hidden(B)
            # for t in range(T):
            #     out, hs = self.agents.critic(state[:, t], actions[:, t], hs)
            #     critic_out.append(out)
            # # shape: [B, T, G, 1]
            # q_taken = torch.stack(critic_out, dim=1)[:, :-1]
            q_taken, _ = self.agents.critic(state.view(
                B*T, self.agents.n_agents, -1), actions.view(B*T, self.agents.n_agents, -1), None)
            q_taken = q_taken.view(B, T, self.agents.n_agents, -1)[:, :-1]

            # all shapes except masks are [B, T, G, 1], and masks are [B, T, 1, 1]
            td_target = rewards[:, :-1] + self.gamma * \
                masks[:, 1:] * target_critic_out[:, 1:]
            td_error = (q_taken - td_target.detach()) ** 2 / 2
            critic_loss = (td_error * masks[:, :-1]
                           ).sum() / masks[:, :-1].sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_params, self.max_grad_norm)
            self.critic_optimizer.step()

            # train actor
            actor_out = []
            hs = self.agents.init_hidden(B)
            for t in range(T):
                out, hs = self.agents.actor(obs[:, t], hs, avail_actions[:, t])
                actor_out.append(out)
            # shape: [B, T, G, action_shape]
            actor_out = torch.stack(actor_out, dim=1)
            if self.action_type == "discrete":
                actor_out = F.gumbel_softmax(actor_out, hard=True)
            if self.action_type == "box":
                actor_out = torch.tanh(actor_out)

            # compute loss
            actor_target = []
            for i in range(self.agents.n_agents):
                joint_actions = actions.detach().clone()
                joint_actions[:, :, i] = actor_out[:, :, i]

                # critic_out = []
                # hs = self.agents.init_hidden(B)
                # for t in range(T - 1):
                #     out, hs = self.agents.critic(
                #         state[:, t], joint_actions[:, t], hs)
                #     # [B, G, 1] -> [B, 1]
                #     critic_out.append(out.mean(dim=1))
                # # append [B, T-1, 1]
                # actor_target.append(torch.stack(critic_out, dim=1))

                critic_out, _ = self.agents.critic(state.view(
                    B*T, self.agents.n_agents, -1), joint_actions.view(B*T, self.agents.n_agents, -1), None)
                actor_target.append(critic_out.view(
                    B, T, self.agents.n_agents, -1)[:, :-1].mean(dim=2))

            # actor_target shape: [B, T-1, 1] x G, actions shape: [B, T, G, action_shape]
            pg_loss = (-torch.stack(actor_target, dim=2) *
                       masks[:, :-1]).sum() / masks[:, :-1].sum()

            # new grad of actor
            # critic_out, _ = self.agents.critic(state.view(
            #     B*T, self.agents.n_agents, -1), actor_out.view(B*T, self.agents.n_agents, -1), None)
            # actor_target = critic_out.view(B, T, self.agents.n_agents, -1)[:, :-1]
            # pg_loss = -(actor_target * masks[:, :-1]).sum() / masks[:, :-1].sum()

            actor_reg = (actor_out[:, :-1] ** 2).mean()
            actor_loss = pg_loss + actor_reg * self.action_reg

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            train_info["actor_grad_norm"] += actor_grad_norm
            train_info["actor_loss"] += actor_loss.item()
            train_info["critic_grad_norm"] += critic_grad_norm
            train_info["critic_loss"] += critic_loss.item()
            train_info["q_taken_mean"] += (q_taken *
                                           masks[:, :-1]).sum() / masks[:, :-1].sum()
            train_info["target_mean"] += (td_target *
                                          masks[:, :-1]).sum() / masks[:, :-1].sum()
            train_info["actions_mean"] += actor_out[:, :-1].mean()
            train_info["actions_norm_mean"] += actor_reg
            

        if self.target_update_hard:
            if episodes - self.last_update_target > self.target_update_interval:
                self.agents.update_targets_hard() 
                self.last_update_target = episodes
        else:
            self.agents.update_targets_soft(self.target_update_interval)

        for k in train_info:
            train_info[k] /= N

        return train_info
