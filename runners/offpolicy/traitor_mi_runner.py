import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.nets.club import CLUBCategorical, CLUBContinuous
from agents.nets.mine import MINE
from data.episode_buffer import EpisodeBatch, EpisodeBuffer
from data.MIbuffer import MIBuffer
from agents import REGISTRY as agent_registry
from learners import REGISTRY as learner_registry


class TraitorMIRunner:
    def __init__(self, args, envs, eval_envs):
        self.args = args
        self.envs = envs
        self.eval_envs = eval_envs

        self.total_timesteps = args.total_timesteps
        self.buffer_size = args.buffer_size
        self.parallel_num = args.parallel_num
        self.episode_limit = args.episode_limit
        self.n_agents = args.n_env_agents
        self.action_shape = args.action_shape

        # init log and model saving
        self.run_dir = args.run_dir
        self.log_dir = os.path.join(self.run_dir, "logs")
        if not self.args.evaluate:
            os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = os.path.join(self.run_dir, "models")
        if not self.args.evaluate:
            os.makedirs(self.model_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)

        print("Observation shape:", args.observation_shape)
        print("State shape:", args.state_shape)
        print("Action shape:", args.action_shape)
        print("Number of agents:", args.n_env_agents)
        print("Adversarial agent ids:", args.adversary_policy.adv_agent_ids)

        # policy
        self.agents = agent_registry[args.victim_policy.agent_name](args.observation_shape, args.state_shape, args.action_shape, args.action_type, args.n_env_agents, device=args.device, **args.victim_policy.agent_args)
        self.agents.prep_rollout()

        # adv policy
        normal_agent_ids = []
        for i in range(self.n_agents):
            if i not in args.adversary_policy.adv_agent_ids:
                normal_agent_ids.append(i)
        self.adv_agent_ids = np.array(args.adversary_policy.adv_agent_ids, dtype=np.int64)
        self.normal_agent_ids = np.array(normal_agent_ids, dtype=np.int64)
        self.n_adv_agents = len(self.adv_agent_ids)
        self.adv_agents = agent_registry[args.adversary_policy.agent_name](args.observation_shape, args.state_shape, args.action_shape, args.action_type, self.n_adv_agents, device=args.device, **args.adversary_policy.agent_args)

        self.mine = MINE(args.state_shape, args.n_env_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
        self.mine_normal = MINE(args.state_shape, (args.n_env_agents - self.n_adv_agents) * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
        self.mine_normal_adv = MINE((args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
        self.mine_normal_state_adv = MINE(args.state_shape + (args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
        if args.action_type == "discrete":
            self.club = CLUBCategorical(args.state_shape, args.n_env_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal = CLUBCategorical(args.state_shape, (args.n_env_agents - self.n_adv_agents) * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal_adv = CLUBCategorical((args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal_state_adv = CLUBCategorical(args.state_shape + (args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)        
        else:
            self.club = CLUBContinuous(args.state_shape, args.n_env_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal = CLUBContinuous(args.state_shape, (args.n_env_agents - self.n_adv_agents) * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal_adv = CLUBContinuous((args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)
            self.club_normal_state_adv = CLUBContinuous(args.state_shape + (args.n_env_agents - self.n_adv_agents) * args.action_shape, self.n_adv_agents * args.action_shape, args.adversary_policy.learner_args.mi_hidden_dim)        

        self.mi_estimator = {'Club': self.club, 'Club_normal': self.club_normal, 'Mine': self.mine, 'Mine_normal': self.mine_normal,
                             'Club_normal_adv': self.club_normal_adv, 'Club_normal_state_adv': self.club_normal_state_adv,
                             'Mine_normal_adv': self.mine_normal_adv, 'Mine_normal_state_adv': self.mine_normal_state_adv}
        self.learner = learner_registry[args.adversary_policy.learner_name](self.adv_agents, mi_estimator=self.mi_estimator, **args.adversary_policy.learner_args)

        # buffer
        self.scheme = {
            "obs": {"vshape": (args.observation_shape,), "group": self.n_adv_agents},
            "state": {"vshape": (args.state_shape,), "group": self.n_adv_agents},
            "avail_actions": {"vshape": (args.action_shape,), "group": self.n_adv_agents},
            "actions": {"vshape": (args.n_actions,), "group": self.n_adv_agents},
            "actions_onehot": {"vshape": (args.action_shape,), "group": self.n_adv_agents},
            "rewards": {"vshape": (1,), "group": self.n_adv_agents},
            "env_rewards": {"vshape": (1,), "group": self.n_adv_agents},
            "rewards_run": {"vshape": (1,), "group": self.n_adv_agents},
            "masks": {"vshape": (1,), "group": 1},
        }
        self.buffer = EpisodeBuffer(
            self.scheme,
            self.buffer_size,
            self.episode_limit + 1
        )

        self.pmic_buffer = MIBuffer(0, args.neg_buffer_size, args.n_env_agents, args.state_shape, args.action_shape)

        # load model
        if args.victim_policy.checkpoint_path:
            print("Load checkpoints from", args.victim_policy.checkpoint_path)
            self.agents.load(args.victim_policy.checkpoint_path)

        if args.adversary_policy.checkpoint_path:
            print("Load adversarial checkpoints from", args.adversary_policy.checkpoint_path)
            self.adv_agents.load(args.adversary_policy.checkpoint_path)

    def reset(self):
        self.batch = EpisodeBatch(
            self.scheme,
            self.parallel_num,
            self.episode_limit + 1
        )

        self.envs.reset()
        obs = self.envs.get_obs()
        state = self.envs.get_state()
        avail_actions = self.envs.get_avail_actions()

        self.batch.add({"obs": obs[:, self.adv_agent_ids], "state": state[:, self.adv_agent_ids], "avail_actions": avail_actions[:, self.adv_agent_ids], "masks": torch.ones((self.parallel_num,))}, t=0)

        return obs, state, avail_actions

    def build_rewards(self):
        self.batch._meta["rewards"] = -self.batch["env_rewards"]
        # if self.args.action_type == "discrete":
        #     self.batch._meta["rewards"] = -self.batch["env_rewards"]
        # elif self.args.action_type == "box":
        #     self.batch._meta["rewards"] = -self.batch["rewards_run"]
        # else:
        #     raise NotImplementedError
    
    def insert_pmic_buffer(self, state_list, actions_list, rewards_list):
        state_tensor = torch.tensor(state_list).float()     # [t, parallel, n, dim]
        rewards = torch.tensor(rewards_list).float()        # [t, parallel, n]
        rewards = torch.sum(rewards, dim=0).mean(-1)
        for i in range(self.parallel_num):
            for t in range(len(state_list)):
                actions_onehot = torch.zeros([self.n_agents, self.action_shape]) 
                for j in range(self.n_agents):
                    if self.args.action_type == "box":
                        actions_onehot[j] = torch.from_numpy(actions_list[t][i][j])
                    elif self.args.action_type == "discrete":
                        actions_onehot[j][actions_list[t][i][j]] = 1
                self.pmic_buffer.add_neg(state_tensor[t][i], actions_onehot, rewards[i])

    def run(self):
        if self.args.evaluate:
            self.eval(0)
            if self.args.save_replay:
                self.eval_envs.save_replay()
            return

        start = time.time()
        
        last_log = 0
        last_save = 0
        last_eval = 0
        global_t = 0
        episodes = 0

        self.eval(0)

        while global_t < self.total_timesteps:
            obs, state, avail_actions = self.reset()
            env_masks = np.ones((self.parallel_num,))
            # Collect
            self.adv_agents.prep_rollout()
            actor_hs = self.agents.init_hidden(self.parallel_num)
            adv_actor_hs = self.adv_agents.init_hidden(self.parallel_num)
            if self.args.victim_policy.agent_name == 'belief':
                belief_hs = self.agents.init_hidden(self.parallel_num)
            states_list, actions_list, rewards_list = [], [], []
            for step in range(self.episode_limit):
                # sample actions
                with torch.no_grad():
                    if self.args.victim_policy.agent_name == 'belief':
                        belief, belief_hs = self.agents.perform_belief(obs, belief_hs)
                        actions, actor_hs = self.agents.perform(obs, belief, actor_hs, avail_actions)
                    else:
                        actions, actor_hs = self.agents.perform(obs, actor_hs, avail_actions)
                adv_actions, adv_actor_hs = self.learner.collect(self.batch, step, global_t, actor_hs=adv_actor_hs)

                actions[:, self.adv_agent_ids] = adv_actions

                cpu_actions = actions.detach().cpu().numpy()
                rewards, dones, infos = self.envs.step(cpu_actions)
                states_list.append(state), actions_list.append(cpu_actions), rewards_list.append(rewards)
                # if self.args.action_type == "box":
                #     rewards_run = []
                #     if "reward_run" in info.keys():
                #         for info in infos:
                #             rewards_run.append([info["reward_run"] for _ in range(self.n_agents)])
                #     rewards_run = np.array(rewards_run)

                masks = 1 - dones
                env_masks = masks * env_masks

                obs = self.envs.get_obs()
                state = self.envs.get_state()
                avail_actions = self.envs.get_avail_actions()

                self.batch.add({"env_rewards": rewards[:, self.adv_agent_ids]}, step)
                # if self.args.action_type == "box":
                #     self.batch.add({"rewards_run": rewards_run[:, self.adv_agent_ids]}, step)
                self.batch.add({"obs": obs[:, self.adv_agent_ids], "state": state[:, self.adv_agent_ids], "avail_actions": avail_actions[:, self.adv_agent_ids], "masks": env_masks,}, step + 1)

                if np.sum(env_masks) == 0:
                    break
            # self.insert_pmic_buffer(states_list, actions_list, rewards_list)
            # get last actions and values (no need to perform a step)
            self.learner.collect(self.batch, step + 1, global_t, actor_hs=adv_actor_hs)

            self.build_rewards()
            self.buffer.insert(self.batch)
        
            # learn
            train_info = {}
            for _ in range(self.args.train_epochs):
                # mi_dict = {}
                # mi_loss_dict = {}
                # for key in self.mi_estimator.keys():
                #     mi_dict[key] = 0.0
                #     mi_loss_dict[key] = 0.0
                # if global_t >= 10000:
                #     mi_loss_dict = self.learner.learn_mi(self.pmic_buffer, self.normal_agent_ids, self.adv_agent_ids)
                #     mi_dict = self.learner.cal_mi(self.pmic_buffer, self.normal_agent_ids, self.adv_agent_ids)   
                if self.args.sample_timestep:
                    if self.buffer.can_sample_timestep(self.args.batch_size, self.args.num_batches):
                        self.adv_agents.prep_training()
                        samples = self.buffer.sample_timestep(self.args.batch_size, self.args.num_batches)
                        train_info = self.learner.learn(samples, episodes)
                else:
                    if self.buffer.can_sample(self.args.batch_size, self.args.num_batches):
                        self.adv_agents.prep_training()
                        samples = self.buffer.sample(self.args.batch_size, self.args.num_batches)
                        train_info = self.learner.learn(samples, episodes)
            
            # post_process
            global_t += (step + 1) * self.parallel_num
            episodes += self.parallel_num
            if global_t - last_save >= self.args.save_interval or global_t >= self.total_timesteps:
                self.adv_agents.save(os.path.join(self.model_dir, str(global_t)))
                last_save = global_t

            # log information
            if global_t - last_log >= self.args.log_interval:
                end = time.time()
                print("Collect at timestep {}/{}, FPS {}".format(global_t, self.total_timesteps, int(global_t / (end - start))))
                self.writer.add_scalar("train/mean_step_rewards", self.buffer["rewards"][:-1].mean().item(), global_t)
                self.writer.add_scalar("train/mean_step_env_rewards", self.buffer["env_rewards"][:-1].mean().item(), global_t)
                # for key in self.mi_estimator.keys():
                #     self.writer.add_scalar("train/mi_loss_" + key, mi_loss_dict[key], global_t)
                #     self.writer.add_scalar("train/mi_" + key, mi_dict[key], global_t)
                for k in train_info:
                    self.writer.add_scalar("train/" + k, train_info[k], global_t)
                last_log = global_t

            # eval
            if global_t - last_eval >= self.args.eval_interval or global_t >= self.total_timesteps:
                self.eval(global_t)
                last_eval = global_t

    def eval(self, total_t):
        log_info = {}
        self.adv_agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            adv_actor_hs = self.adv_agents.init_hidden(1)
            if self.args.victim_policy.agent_name == 'belief':
                belief_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": [], "return_total": []}
            for step in range(self.args.episode_limit):
                with torch.no_grad():
                    if self.args.victim_policy.agent_name == 'belief':
                        belief, belief_hs = self.agents.perform_belief(obs, belief_hs)
                        actions, actor_hs = self.agents.perform(obs, belief, actor_hs, avail_actions)
                    else:
                        actions, actor_hs = self.agents.perform(obs, actor_hs, avail_actions)
                    adv_actions, adv_actor_hs = self.adv_agents.perform(obs[:, self.adv_agent_ids], adv_actor_hs, avail_actions[:, self.adv_agent_ids])
                    
                # adv_actions[:, :, 1] = adv_actions[:, :, 0]
                actions[:, self.adv_agent_ids] = adv_actions
                # print(actions[:, 0])
                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                episode_info["return"].append(rewards[0][0])
                episode_info["return_total"].append(sum(rewards[0]))
                for key in infos[0]:
                    if key not in episode_info:
                        episode_info[key] = []
                    episode_info[key].append(infos[0][key])

                obs = self.eval_envs.get_obs()
                avail_actions = self.eval_envs.get_avail_actions()

                if np.all(dones):
                    episode_info["ep_length"].append(step + 1)
                    break
                    
            for key in episode_info:
                if key not in log_info:
                    log_info[key] = []
                log_info[key].append(episode_info[key])

        print(f"Eval for {self.args.eval_episodes} episodes at timestep {total_t}")
        for key in log_info:
            if key in self.args.sum_keys:
                value = np.mean([np.sum(elm) for elm in log_info[key]])
                if not self.args.evaluate:
                    self.writer.add_scalar(f"test/{key}_mean", value, total_t)
                print("Mean of {}: {:.4f}".format(key, value))
            elif key in self.args.mean_keys:
                value = np.mean([np.mean(elm) for elm in log_info[key]])
                if not self.args.evaluate:
                    self.writer.add_scalar(f"test/{key}_mean", value, total_t)
                print("Mean of {}: {:.4f}".format(key, value))
            elif key in self.args.last_keys:
                value = np.mean([elm[-1] for elm in log_info[key]])
                if not self.args.evaluate:
                    self.writer.add_scalar(f"test/{key}_mean", value, total_t)
                print("Mean of {}: {:.4f}".format(key, value))
