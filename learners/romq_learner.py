import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_learner import BaseLearner
from agents.nets.transition_model import TransitionModel
from agents.nets.mine import MINE
from agents.nets.club import CLUBCategorical, CLUBContinuous


def onehot_from_logits(logits):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    return argmax_acs

def minq_onehot_from_logits(logits):
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.min(-1, keepdim=True)[0]).float()
    return argmax_acs


class ROMQLearner(BaseLearner):
    def __init__(self, agents, noise_scale=0.1, start_steps=10000, gamma=0.99, use_adam=True, actor_lr=5e-4,
                 critic_lr=5e-4, optim_alpha=0.99, optim_eps=1e-5, max_grad_norm=10, action_reg=0.001,
                 target_update_hard=False, target_update_interval=0.01, state_prob=1/3, state_epsilon=0.1, 
                 state_iter=10, state_alpha=0.02, action_prob=1/3, action_epsilon=1, action_iter=10, 
                 action_alpha=0.2, use_mi=False, transition_hidden_dim=256, transition_lr=1e-3, 
                 transition_epochs=5, mi_hidden_dim=256, mi_lr=1e-3, mi_epochs=5, mine_coef=0.1, club_coef=0.1):
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
        self.state_prob = state_prob
        self.state_epsilon = state_epsilon
        self.state_iter = state_iter
        self.state_alpha = state_alpha
        self.action_prob = action_prob
        self.action_epsilon = action_epsilon
        # self.action_iter = action_iter
        self.action_iter = action_iter
        self.action_alpha = action_alpha
        self.use_mi = use_mi
        self.transition_epochs = transition_epochs
        self.mi_epoches = mi_epochs
        self.episode_adversary = 0
        self.agent_adversary = 0
        self.mine_coef = mine_coef
        self.club_coef = club_coef

        if self.use_mi:
            self.transition_model = TransitionModel(agents.obs_shape, agents.action_shape, agents.n_agents, transition_hidden_dim, False).to(self.device)
            self.transition_params = list(self.transition_model.parameters())

            self.mine = MINE(agents.obs_shape, agents.action_shape, mi_hidden_dim).to(self.device)
            self.mine_params = list(self.mine.parameters())
            if self.action_type == "discrete":
                self.club = CLUBCategorical(agents.obs_shape, agents.action_shape, mi_hidden_dim).to(self.device)
            else:
                self.club = CLUBContinuous(agents.obs_shape, agents.action_shape, mi_hidden_dim).to(self.device)
            self.club_params = list(self.club.parameters())

        self.actor_params = list(self.agents.actor.parameters())
        self.critic_params = list(self.agents.critic.parameters())
        self.minq_params = list(self.agents.minq_net.parameters())
        if use_adam:
            self.actor_optimizer = torch.optim.Adam(
                self.agents.actor.parameters(), lr=actor_lr, eps=optim_eps)
            self.critic_optimizer = torch.optim.Adam(
                self.agents.critic.parameters(), lr=critic_lr, eps=optim_eps)
            self.minq_optimizer = torch.optim.Adam(
                self.agents.minq_net.parameters(), lr=critic_lr, eps=optim_eps)
            if self.use_mi:
                self.transition_optimizer = torch.optim.Adam(
                    self.transition_model.parameters(), lr=transition_lr, eps=optim_eps)
                self.mine_optimizer = torch.optim.Adam(
                    self.mine.parameters(), lr=mi_lr, eps=optim_eps)
                self.club_optimizer = torch.optim.Adam(
                    self.club.parameters(), lr=mi_lr, eps=optim_eps)
        else:
            self.actor_optimizer = torch.optim.RMSprop(
                self.agents.actor.parameters(), lr=actor_lr, alpha=optim_alpha, eps=optim_eps)
            self.critic_optimizer = torch.optim.RMSprop(
                self.agents.critic.parameters(), lr=critic_lr, alpha=optim_alpha, eps=optim_eps)
            self.minq_optimizer = torch.optim.RMSprop(
                self.agents.minq_net.parameters(), lr=critic_lr, alpha=optim_alpha, eps=optim_eps)
            if self.use_mi:
                self.transition_optimizer = torch.optim.RMSprop(
                    self.transition_model.parameters(), lr=transition_lr, alpha=optim_alpha, eps=optim_eps)
                self.mine_optimizer = torch.optim.RMSprop(
                    self.mine.parameters(), lr=mi_lr, alpha=optim_alpha, eps=optim_eps)
                self.club_optimizer = torch.optim.RMSprop(
                    self.club.parameters(), lr=mi_lr, alpha=optim_alpha, eps=optim_eps)

    def collect_noise_obs(self, obs_ori, actor_hs, avail_actions, state):
        # obs_ori: [N, G, obs_shape]
        obs = obs_ori.detach().clone()
        for i in range(self.state_iter):
            obs.requires_grad_(True)
            actions, hs = self.agents.actor(obs, actor_hs, avail_actions)
            if self.action_type == "discrete":
                actions_onehot = F.gumbel_softmax(actions, hard=True)
            elif self.action_type == "box":
                actions_onehot = torch.tanh(actions)
            q_value, _ = self.agents.critic(state, actions_onehot, None)
            q_value.sum().backward()
            obs_grad = obs.grad.sign()[:, self.agent_adversary]
            obs = obs.detach()
            obs[:, self.agent_adversary] -= self.state_alpha * obs_grad
            obs = (torch.clamp(obs - obs_ori, -self.state_epsilon, self.state_epsilon) + obs_ori).detach()
        return obs

    # def collect_noise_actions(self, actions_ori, state):
    #     # actions_ori: [N, G, actions_shape]
    #     actions = actions_ori.detach().clone()
    #     for _ in range(self.action_iter):
    #         actions.requires_grad_(True)
    #         if self.action_type == "discrete":
    #             actions_onehot = F.gumbel_softmax(actions, hard=True)
    #         elif self.action_type == "box":
    #             actions_onehot = torch.tanh(actions)
    #         q_value, _ = self.agents.critic(state, actions_onehot, None)
    #         q_value.sum().backward()
    #         action_grad = actions.grad.sign()[:, self.agent_adversary]
    #         actions = actions.detach()
    #         actions[:, self.agent_adversary] -= self.action_alpha * action_grad
    #         actions = (torch.clamp(actions - actions_ori, -self.action_epsilon, self.action_epsilon) + actions_ori).detach()
    #     return actions

    def collect_noise_actions(self, actions_ori, state):
        # actions_ori: [N, G, actions_shape]
        actions_onehot_ori = torch.tanh(actions_ori)
        actions_onehot = actions_onehot_ori.detach().clone()
        for _ in range(self.action_iter):
            actions_onehot.requires_grad_(True)
            q_value, _ = self.agents.critic(state, actions_onehot, None)
            q_value.sum().backward()
            action_grad = actions_onehot.grad.sign()[:, self.agent_adversary]
            actions_onehot = actions_onehot.detach()
            actions_onehot[:, self.agent_adversary] -= self.action_alpha * action_grad
            actions_onehot = (torch.clamp(actions_onehot - actions_onehot_ori, -self.action_epsilon, self.action_epsilon) + actions_onehot_ori).detach()
            actions_onehot = torch.clamp(actions_onehot, -1, 1)
        return actions_onehot

    def collect_minq_actions(self, actions_ori, avail_actions, state):
        # actions_ori: [N, G, actions_shape]
        actions_shape = avail_actions.shape[-1]
        actions_onehot = onehot_from_logits(actions_ori)
        q_values = []
        for i in range(actions_shape):
            actions = actions_onehot.detach().clone()
            actions[:, self.agent_adversary] = 0
            actions[:, self.agent_adversary, i] = 1
            q_value, _ = self.agents.critic(state, actions, None)
            q_values.append(q_value)
        q_values = torch.cat(q_values, dim=2).sum(dim=1) # [N, actions_shape]
        q_values[avail_actions[:, self.agent_adversary] == 0] = 1e10
        # print(q_values)
        actions = F.one_hot(q_values.argmin(dim=-1).detach(), num_classes=actions_shape) # [N, actions_shape]
        actions[actions==0] = -1e10

        actions_ori = actions_ori.detach().clone()
        actions_ori[:, self.agent_adversary] = actions

        return actions_ori

    def collect_minq_net_actions(self, actions_ori, avail_actions, obs):
        # actions_ori: [N, G, actions_shape]
        actions_shape = avail_actions.shape[-1]

        q_values, _ = self.agents.minq_net(obs[:, self.agent_adversary], None)
        q_values[avail_actions[:, self.agent_adversary] == 0] = 1e10

        actions = F.one_hot(q_values.argmin(dim=-1).detach(), num_classes=actions_shape) # [N, actions_shape]
        actions[actions==0] = -1e10

        actions_ori = actions_ori.detach().clone()
        actions_ori[:, self.agent_adversary] = actions

        return actions_ori

    def collect(self, batch, t, global_t, actor_hs=None):
        obs = batch["obs"][:, t].to(self.device)
        noise_obs = obs.detach().clone()
        state = batch["state"][:, t].to(self.device)
        avail_actions = batch["avail_actions"][:, t].to(self.device)
        noise_type = torch.zeros((obs.shape[0], obs.shape[1]))
        if global_t > self.start_steps:
            onehot = False
            # when t==0, generate a random number to decide the adversary
            if t == 0:
                prob = torch.rand(1).item()
                if prob < self.state_prob:
                    self.episode_adversary = 1
                elif prob - self.state_prob < self.action_prob:
                    self.episode_adversary = 2
                else:
                    self.episode_adversary = 0
                self.agent_adversary = torch.randint(self.agents.n_agents, (1,)).item()
            if self.episode_adversary == 0:
                # episode_adversary==0, no adversary
                actions, hs = self.agents.actor(obs, actor_hs, avail_actions)
                cf_actions = actions.clone()
            if self.episode_adversary == 1:
                # episode_adversary==1, state adversary
                cf_actions, _ = self.agents.actor(obs, actor_hs, avail_actions)
                noise_obs = self.collect_noise_obs(obs, actor_hs, avail_actions, state)
                actions, hs = self.agents.actor(noise_obs, actor_hs, avail_actions)
                # noise_type: adversary=o'-o, other=o-T(o). when t=0, T(o) is null, so use noise0.
                if t > 0:
                    noise_type[:] = 1
                    noise_type[:, self.agent_adversary] = 0
            elif self.episode_adversary == 2:
                # episode_adversary==2, action adversary
                actions, hs = self.agents.actor(obs, actor_hs, avail_actions)
                cf_actions = actions.clone()
                if self.action_type == "discrete":
                    # actions = self.collect_minq_actions(actions, avail_actions, state)
                    actions = self.collect_minq_net_actions(actions, avail_actions, obs)
                elif self.action_type == "box":
                    actions = self.collect_noise_actions(actions, state)
                    onehot = True
                else:
                    raise NotImplementedError
                # noise_type: adversary=0, other=o-T(o). when t=0, T(o) is null, so use noise0.
                if t > 0:
                    noise_type[:] = 1
                    noise_type[:, self.agent_adversary] = 0

            if self.action_type == "discrete":
                actions_onehot = F.gumbel_softmax(actions, hard=True)
                actions = actions_onehot.argmax(dim=-1)
                cf_actions_onehot = F.gumbel_softmax(cf_actions, hard=True)
            elif self.action_type == "box":
                if not onehot:
                    actions = torch.tanh(actions)
                actions += torch.randn_like(actions) * self.noise_scale
                actions = torch.clamp(actions, -1, 1)
                actions_onehot = actions.clone()
                cf_actions_onehot = cf_actions.clone()
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
            cf_actions_onehot = actions_onehot.clone()

        episode_adversarys = torch.ones([batch.size, batch.scheme["episode_adversary"]["group"], *batch.scheme["episode_adversary"]["vshape"]]) * self.episode_adversary
        agent_adversarys = torch.ones([batch.size, batch.scheme["agent_adversary"]["group"], *batch.scheme["agent_adversary"]["vshape"]]) * self.agent_adversary
        batch.add({"actions": actions, "actions_onehot": actions_onehot, "noise_obs": noise_obs,
            "cf_actions_onehot": cf_actions_onehot, "episode_adversary": episode_adversarys, 
            "agent_adversary": agent_adversarys, "noise_type": noise_type}, t)

        if hs is not None:
            hs = hs.detach()
        return actions, hs

    def update_transition_model(self, samples, train_info):
        obs = samples["obs"].to(self.device)
        actions = samples["actions_onehot"].to(self.device)
        masks = samples["masks"].to(self.device)

        # [B, T, G, ...]
        for _ in range(self.transition_epochs):
            obs_pred = []
            for t in range(obs.shape[1] - 1):
                out, _ = self.transition_model(obs[:, t], actions[:, t], None)
                obs_pred.append(out)
            obs_pred = torch.stack(obs_pred, dim=1)

            distance = (obs_pred - obs[:, 1:]) ** 2 / 2
            loss = (distance * masks[:, :-1]).sum() / masks[:, :-1].sum()

            self.transition_optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.transition_params, self.max_grad_norm)
            self.transition_optimizer.step()

        train_info["transition_loss"] += loss.item()
        train_info["transition_grad_norm"] += grad_norm

    def update_mi_model(self, samples, train_info):
        B, T = samples["filled"].shape

        obs_prev = samples["obs/t-1"].to(self.device)
        obs = samples["obs"].to(self.device)
        noise_obs = samples["noise_obs"].to(self.device)
        actions = samples["actions_onehot"].to(self.device)
        cf_actions_prev = samples["cf_actions_onehot/t-1"].to(self.device)
        noise_type = samples["noise_type"].to(self.device)
        masks = samples["masks"].to(self.device)

        # calculate T(o_t, a_t)
        obs_approx = []
        for t in range(T):
            out, _ = self.transition_model(obs_prev[:, t], cf_actions_prev[:, t], None)
            obs_approx.append(out)
        obs_approx = torch.stack(obs_approx, dim=1).detach()

        # calculate N_t
        noise0 = noise_obs - obs
        noise1 = noise_obs - obs_approx
        noise = torch.where(noise_type== 1, noise1, noise0).detach()

        for _ in range(self.mi_epoches):
            # learn MINE [B*T*G, shape] -> 1
            mine_loss = []
            for t in range(T):
                mine_loss.append(self.mine.learning_loss(
                    obs_approx[:, t].reshape(-1, self.agents.obs_shape), 
                    actions[:, t].reshape(-1, self.agents.action_shape)
                    ).reshape(B, self.agents.n_agents, 1))
            mine_loss = torch.stack(mine_loss, dim=1)
            mine_loss = (mine_loss * masks).sum() / masks.sum()
            self.mine_optimizer.zero_grad()
            mine_loss.backward()
            mine_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.mine_params, self.max_grad_norm)
            self.mine_optimizer.step()

            # learn CLUB [B*T*G, shape] -> 1
            club_loss = []
            for t in range(T):
                club_loss.append(self.club.learning_loss(
                    noise[:, t].reshape(-1, self.agents.obs_shape),
                    actions[:, t].reshape(-1, self.agents.action_shape).long()
                    ).reshape(B, self.agents.n_agents, 1))
            club_loss = torch.stack(club_loss, dim=1)
            club_loss = (club_loss * masks).sum() / masks.sum()
            self.club_optimizer.zero_grad()
            club_loss.backward()
            club_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.club_params, self.max_grad_norm)
            self.club_optimizer.step()

        train_info["mine_loss"] += mine_loss.item()
        train_info["mine_grad_norm"] += mine_grad_norm
        train_info["club_loss"] += club_loss.item()
        train_info["club_grad_norm"] += club_grad_norm

        with torch.no_grad():
            mine_mi = []
            for t in range(T):
                mine_mi.append(self.mine(
                    obs_approx[:, t].reshape(-1, self.agents.obs_shape), 
                    actions[:, t].reshape(-1, self.agents.action_shape)
                    ).reshape(B, self.agents.n_agents, 1))
            mine_mi = torch.stack(mine_mi, dim=1)

            club_mi = []
            for t in range(T):
                club_mi.append(self.club(
                    noise[:, t].reshape(-1, self.agents.obs_shape),
                    actions[:, t].reshape(-1, self.agents.action_shape)
                    ).reshape(B, self.agents.n_agents, 1))
            club_mi = torch.stack(club_mi, dim=1)

        return mine_mi.detach().cpu(), club_mi.detach().cpu()

    def update_maddpg(self, samples, train_info):
        B, T = samples["filled"].shape

        # [B, T, G, V]
        obs = samples["obs"].to(self.device)
        state = samples["state"].to(self.device)
        actions = samples["actions_onehot"].to(self.device)
        avail_actions = samples["avail_actions"].to(self.device)
        rewards = samples["rewards"].to(self.device)
        # [B, T, G, 1]
        masks = samples["masks"].to(self.device).expand_as(rewards)
        episode_adversary = samples["episode_adversary"].to(self.device).expand_as(rewards).long()
        # [B, T, 1, 1]
        agent_adversary = samples["agent_adversary"].to(self.device).squeeze(-1).squeeze(-1).long()

        # when episode_adversary=2 (actions), the actor update should be masked
        actor_mask = 1 - ((episode_adversary == 2) * F.one_hot(agent_adversary, num_classes=rewards.shape[2]).unsqueeze(-1))
        actor_mask = actor_mask * masks

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

        # compute loss
        actor_target = []
        for i in range(self.agents.n_agents):
            joint_actions = actions.detach().clone()
            joint_actions[:, :, i] = actor_out[:, :, i]

            critic_out, _ = self.agents.critic(state.view(
                B*T, self.agents.n_agents, -1), joint_actions.view(B*T, self.agents.n_agents, -1), None)
            actor_target.append(critic_out.view(
                B, T, self.agents.n_agents, -1)[:, :-1].mean(dim=2))

        # actor_target shape: [B, T-1, 1] x G, actions shape: [B, T, G, action_shape]
        pg_loss = (-torch.stack(actor_target, dim=2) *
                    actor_mask[:, :-1]).sum() / actor_mask[:, :-1].sum()
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

    def update_minq_net(self, samples, train_info):
        B, T = samples["filled"].shape

        # [B, T, G, V]
        obs = samples["obs"].to(self.device)
        state = samples["state"].to(self.device)
        actions = samples["actions"].to(self.device)
        avail_actions = samples["avail_actions"].to(self.device)
        rewards = samples["rewards"].to(self.device)
        # [B, T, G, 1]
        masks = samples["masks"].to(self.device).expand_as(rewards)
        episode_adversary = samples["episode_adversary"].to(self.device).expand_as(rewards).long()
        # [B, T, 1, 1]
        agent_adversary = samples["agent_adversary"].to(self.device).squeeze(-1).squeeze(-1).long()

        actor_mask = ((episode_adversary == 2) * F.one_hot(agent_adversary, num_classes=rewards.shape[2]).unsqueeze(-1))
        actor_mask = actor_mask * masks

        q_values = []
        for t in range(T):
            # q_value, _ = self.agents.minq_net(obs[:, t], None, avail_actions[:, t])
            q_value, _ = self.agents.minq_net(obs[:, t], None)
            q_values.append(q_value)
        q_values = torch.stack(q_values, dim=1)
        chosen_q_values = torch.gather(q_values, dim=-1, index=actions.long())

        target_q_values = []
        for t in range(T):
            # target_q_value, _ = self.agents.target_minq_net(obs[:, t], None, avail_actions[:, t])
            target_q_value, _ = self.agents.target_minq_net(obs[:, t], None)
            target_q_values.append(target_q_value)
        target_q_values = torch.stack(target_q_values, dim=1).max(dim=-1)[0].unsqueeze(-1)

        targets = rewards[:, :-1] + self.gamma * target_q_values[:, 1:] * masks[:, 1:]
        td_error = (chosen_q_values[:, :-1] - targets.detach()) ** 2 / 2

        minq_loss = (td_error * masks[:, :-1]).sum() / masks[:, :-1].sum()

        self.minq_optimizer.zero_grad()
        minq_loss.backward()
        minq_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.minq_params, self.max_grad_norm)
        self.minq_optimizer.step()

        train_info["minq_loss"] += minq_loss.item()
        train_info["minq_grad_norm"] += minq_grad_norm

    
    def learn(self, samples, episodes):
        train_info = {
            "critic_loss": 0,
            "critic_grad_norm": 0,
            "actor_loss": 0,
            "actor_grad_norm": 0,
            "minq_loss": 0,
            "minq_grad_norm": 0,
            "q_taken_mean": 0,
            "target_mean": 0,
            "actions_mean": 0,
            "actions_norm_mean": 0,
            "transition_loss": 0,
            "transition_grad_norm": 0,
            "mine_loss": 0,
            "mine_grad_norm": 0,
            "club_loss": 0,
            "club_grad_norm": 0,
        }
        # shape: [N, B, T, G, V]
        N = samples["filled"].shape[0]
        for n in range(N):
            samples_n = {k: samples[k][n] for k in samples}
            if self.use_mi:
                # update transition model
                self.update_transition_model(samples_n, train_info)
                # update mutual information estimator
                mine_mi, club_mi = self.update_mi_model(samples_n, train_info)
                # add mutual information [B, T, G, 1] into reward
                samples_n["rewards"] = samples_n["rewards"] + self.mine_coef * mine_mi - self.club_coef * club_mi
            # update algorithm
            if self.action_type == "discrete":
                self.update_minq_net(samples_n, train_info)
            self.update_maddpg(samples_n, train_info)

        if self.target_update_hard:
            if episodes - self.last_update_target > self.target_update_interval:
                self.agents.update_targets_hard()
                self.last_update_target = episodes
        else:
            self.agents.update_targets_soft(self.target_update_interval)

        for k in train_info:
            train_info[k] /= N

        return train_info
