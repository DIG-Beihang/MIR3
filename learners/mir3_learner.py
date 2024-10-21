import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_learner import BaseLearner


def onehot_from_logits(logits): 
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs

 
class MIR3Learner(BaseLearner):
    def __init__(self, agents, param_club, noise_scale=0.1, start_steps=10000, gamma=0.99, use_adam=True, actor_lr=5e-4,
                 critic_lr=5e-4, pmic_lr=5e-4, optim_alpha=0.99, optim_eps=1e-5, max_grad_norm=10, action_reg=0.001,
                 target_update_hard=False, target_update_interval=0.01):
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

        self.param_mine = 0
        self.param_club = param_club

        self.actor_params = list(self.agents.actor.parameters())
        self.critic_params = list(self.agents.critic.parameters())
        self.club_params = list(self.agents.club.parameters())
        self.mine_params = list(self.agents.mine.parameters())
        
        if use_adam:
            self.actor_optimizer = torch.optim.Adam(
                self.agents.actor.parameters(), lr=actor_lr, eps=optim_eps)
            self.critic_optimizer = torch.optim.Adam(
                self.agents.critic.parameters(), lr=critic_lr, eps=optim_eps)
            self.club_optimizer = torch.optim.Adam(
                self.agents.club.parameters(), lr=pmic_lr, eps=optim_eps)
            self.mine_optimizer = torch.optim.Adam(
                self.agents.mine.parameters(), lr=pmic_lr, eps=optim_eps)
        else:
            self.actor_optimizer = torch.optim.RMSprop(
                self.agents.actor.parameters(), lr=actor_lr, alpha=optim_alpha, eps=optim_eps)
            self.critic_optimizer = torch.optim.RMSprop(
                self.agents.critic.parameters(), lr=critic_lr, alpha=optim_alpha, eps=optim_eps)
            self.club_optimizer = torch.optim.RMSprop(
                self.agents.club.parameters(), lr=pmic_lr, alpha=optim_alpha, eps=optim_eps)
            self.mine_optimizer = torch.optim.RMSprop(
                self.agents.mine.parameters(), lr=pmic_lr, alpha=optim_alpha, eps=optim_eps)

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
    
    def learn_club(self, buffer, epochs=1, batch_size=64):
        club_loss_list =[]
        for _ in range(epochs):
            club_state, club_action = buffer.sample_neg(batch_size)
            club_state = club_state.mean(1).to(self.device) # [batch, n_state]
            club_action = club_action.reshape(batch_size, -1).to(self.device) # [batch, n_agent * n_action]
            
            club_loss = self.agents.club.learning_loss(club_state, club_action)
            club_loss_list.append(club_loss.item())
            
            self.club_optimizer.zero_grad()
            club_loss.backward()
            self.club_optimizer.step()
        return sum(club_loss_list) / len(club_loss_list)
    
    def learn_mine(self, buffer, epochs=1, batch_size=64):
        mine_loss_list =[]
        for _ in range(epochs):
            mine_state, mine_action = buffer.sample_neg(batch_size)
            mine_state = mine_state.mean(1).to(self.device) # [batch, n_state]
            mine_action = mine_action.reshape(batch_size, -1).to(self.device) # [batch, n_agent * n_action]
            
            mine_loss = self.agents.mine.learning_loss(mine_state, mine_action)
            mine_loss_list.append(mine_loss.item())
            
            self.mine_optimizer.zero_grad()
            mine_loss.backward()
            self.mine_optimizer.step()
        return sum(mine_loss_list) / len(mine_loss_list)
    
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


    def learn_with_pmic(self, samples, episodes):
        train_info = {
            "critic_loss": 0,
            "critic_grad_norm": 0,
            "actor_loss": 0,
            "actor_grad_norm": 0,
            "q_taken_mean": 0,
            "target_mean": 0,
            "actions_mean": 0,
            "actions_norm_mean": 0,
            "min_upper_bound": 0,
            "lower_bound": 0
        }
        # shape: [N, B, T, G, V]
        # [n_env, batch, timestamp, n_agent, xxx]
        N, B, T = samples["filled"].shape
        for n in range(N):
            obs = samples["obs"][n].to(self.device)
            state = samples["state"][n].to(self.device)
            actions = samples["actions_onehot"][n].to(self.device)
            avail_actions = samples["avail_actions"][n].to(self.device)
            rewards = samples["rewards"][n].to(self.device)
            masks = samples["masks"][n].to(self.device).expand_as(rewards)

            # compute target actions in t+1
            target_actor_out = []
            target_hs = self.agents.init_hidden(B)
            for t in range(T):
                target_out, target_hs = self.agents.target_actor(obs[:, t], target_hs, avail_actions[:, t])
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
            
            target_critic_out, _ = self.agents.target_critic(state.view(B*T, self.agents.n_agents, -1), 
                                                             target_actor_out.view(B*T, self.agents.n_agents, -1), None)
            target_critic_out = target_critic_out.view(B, T, self.agents.n_agents, -1)

            # compute q-values in t
            # critic_out = []
            # hs = self.agents.init_hidden(B)
            # for t in range(T):
            #     out, hs = self.agents.critic(state[:, t], actions[:, t], hs)
            #     critic_out.append(out)
            # # shape: [B, T, G, 1]
            # q_taken = torch.stack(critic_out, dim=1)[:, :-1]
            
            q_taken, _ = self.agents.critic(state.view(B*T, self.agents.n_agents, -1), 
                                            actions.view(B*T, self.agents.n_agents, -1), None)
            q_taken = q_taken.view(B, T, self.agents.n_agents, -1)[:, :-1]
            
            # INPUT: state: [batch, timestamp, n_agent, n_state], actions:[batch, timestamp, n_agent, 1]
            # OUTPUT: [batch, timestamp, 1, 1]
            with torch.no_grad():
                mi_upper_bound = self.agents.club(state.mean(-2).view(B*T, -1), 
                                                      actions.view(B*T, -1))
                mi_upper_bound = mi_upper_bound.view(B, T, 1, -1)

                lower_bound = self.agents.mine(state.mean(-2).view(B*T, -1), 
                                                      actions.view(B*T, -1))
                lower_bound = lower_bound.view(B, T, 1, -1)
                

            # all shapes except masks are [B, T, G, 1], and masks are [B, T, 1, 1]
            td_target = rewards[:, :-1] + self.gamma * masks[:, 1:] * target_critic_out[:, 1:] \
                 - self.param_club * mi_upper_bound[:, :-1]
            td_error = (q_taken - td_target.detach()) ** 2 / 2
            critic_loss = (td_error * masks[:, :-1]).sum() / masks[:, :-1].sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
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
                
                critic_out, _ = self.agents.critic(state.view(B*T, self.agents.n_agents, -1), 
                                                   joint_actions.view(B*T, self.agents.n_agents, -1), None)
                actor_target.append(critic_out.view(B, T, self.agents.n_agents, -1)[:, :-1].mean(dim=2))

            # actor_target shape: [B, T-1, 1] x G, actions shape: [B, T, G, action_shape]            
            pg_loss = (-torch.stack(actor_target, dim=2) * masks[:, :-1]).sum() / masks[:, :-1].sum()
            
            actor_reg = (actor_out[:, :-1] ** 2).mean()
            actor_loss = pg_loss + actor_reg * self.action_reg

            min_upper_bound = self.agents.club(state.mean(-2).view(B*T, -1), 
                                                   actions.view(B*T, -1))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            train_info["actor_grad_norm"] += actor_grad_norm
            train_info["actor_loss"] += actor_loss.item()
            train_info["critic_grad_norm"] += critic_grad_norm
            train_info["critic_loss"] += critic_loss.item()
            
            train_info["min_upper_bound"] += min_upper_bound.mean().item()
            train_info["lower_bound"] += lower_bound.mean().item()
            
            train_info["q_taken_mean"] += (q_taken * masks[:, :-1]).sum() / masks[:, :-1].sum()
            train_info["target_mean"] += (td_target * masks[:, :-1]).sum() / masks[:, :-1].sum()
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

            # compute target actions in t+1
            target_actor_out = []
            target_hs = self.agents.init_hidden(B)
            for t in range(T):
                target_out, target_hs = self.agents.target_actor(obs[:, t], target_hs, avail_actions[:, t])
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
            target_critic_out = target_critic_out.view(B, T, self.agents.n_agents, -1)

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
            td_target = rewards[:, :-1] + self.gamma * masks[:, 1:] * target_critic_out[:, 1:]
            td_error = (q_taken - td_target.detach()) ** 2 / 2
            critic_loss = (td_error * masks[:, :-1]).sum() / masks[:, :-1].sum()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
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
                actor_target.append(critic_out.view(B, T, self.agents.n_agents, -1)[:, :-1].mean(dim=2))

            # actor_target shape: [B, T-1, 1] x G, actions shape: [B, T, G, action_shape]
            pg_loss = (-torch.stack(actor_target, dim=2) * masks[:, :-1]).sum() / masks[:, :-1].sum()
            actor_reg = (actor_out[:, :-1] ** 2).mean()
            actor_loss = pg_loss + actor_reg * self.action_reg

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
            self.actor_optimizer.step()

            train_info["actor_grad_norm"] += actor_grad_norm
            train_info["actor_loss"] += actor_loss.item()
            train_info["critic_grad_norm"] += critic_grad_norm
            train_info["critic_loss"] += critic_loss.item()
            train_info["q_taken_mean"] += (q_taken * masks[:, :-1]).sum() / masks[:, :-1].sum()
            train_info["target_mean"] += (td_target * masks[:, :-1]).sum() / masks[:, :-1].sum()
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