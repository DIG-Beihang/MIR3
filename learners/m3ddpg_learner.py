import torch
import torch.nn as nn
import torch.nn.functional as F

from .maddpg_learner import MADDPGLearner


def onehot_from_logits(logits):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs


class M3DDPGLearner(MADDPGLearner):
    def __init__(self, agents, noise_scale=0.1, start_steps=10000, gamma=0.99, use_adam=True, actor_lr=5e-4,
                 critic_lr=5e-4, optim_alpha=0.99, optim_eps=1e-5, max_grad_norm=10, action_reg=0.001,
                 target_update_hard=False, target_update_interval=0.01, adv_epsilon=0.1):
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
        self.adv_epsilon = adv_epsilon
        self.last_update_target = 0

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

    def learn(self, samples, episodes, club=None):
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
        if club:
            train_info["min_upper_bound"] = 0
        # shape: [N, B, T, G, V]
        N, B, T = samples["filled"].shape
        for n in range(N):
            obs = samples["obs"][n].to(self.device)
            state = samples["state"][n].to(self.device)
            actions = samples["actions_onehot"][n].to(self.device)
            avail_actions = samples["avail_actions"][n].to(self.device)
            rewards = samples["rewards"][n].to(self.device)
            masks = samples["masks"][n].to(self.device).expand_as(rewards)

            for i in range(self.agents.n_agents):
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

                # get adversarial noise
                target_action_noise = self.get_action_noise(self.agents.target_critic, state, target_actor_out)
                masked_action_noise = torch.zeros_like(target_action_noise)
                masked_action_noise[:, :, i] = 1
                masked_action_noise = (target_action_noise * masked_action_noise).detach()
                # masked_action_noise shape: [B, T, G, action_shape]

                # compute target q-values in t+1
                target_actor_out -= masked_action_noise * self.adv_epsilon
                target_critic_out, _ = self.agents.target_critic(state.view(
                    B*T, self.agents.n_agents, -1), target_actor_out.view(B*T, self.agents.n_agents, -1), None)
                target_critic_out = target_critic_out.view(
                    B, T, self.agents.n_agents, -1)

                # compute q-values in t
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

                # get adversarial noise
                action_noise = self.get_action_noise(self.agents.critic, state, actor_out)
                # action_noise shape: [B, T, G, action_shape]

                # compute loss
                actor_target = []
                joint_actions = (actions - self.adv_epsilon * action_noise).detach()
                joint_actions[:, :, i] = actor_out[:, :, i]

                critic_out, _ = self.agents.critic(state.view(
                    B*T, self.agents.n_agents, -1), joint_actions.view(B*T, self.agents.n_agents, -1), None)
                actor_target.append(critic_out.view(
                    B, T, self.agents.n_agents, -1)[:, :-1].mean(dim=2))

                # actor_target shape: [B, T-1, 1] x G, actions shape: [B, T, G, action_shape]
                pg_loss = (-torch.stack(actor_target, dim=2) *
                        masks[:, :-1]).sum() / masks[:, :-1].sum()
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

            if club:
                min_upper_bound = club(state.mean(-2).view(B*T, -1), 
                                                   actions.view(B*T, -1))
                train_info["min_upper_bound"] += min_upper_bound.mean().item()

        if self.target_update_hard:
            if episodes - self.last_update_target > self.target_update_interval:
                self.agents.update_targets_hard()
                self.last_update_target = episodes
        else:
            self.agents.update_targets_soft(self.target_update_interval)

        for k in train_info:
            train_info[k] /= N * self.agents.n_agents

        return train_info
