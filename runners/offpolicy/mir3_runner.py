import os
import time
import torch
import nni
import numpy as np
from torch.utils.tensorboard import SummaryWriter 

from data.episode_buffer import EpisodeBatch, EpisodeBuffer
from data.MIbuffer import MIBuffer
from agents import REGISTRY as agent_registry
from learners import REGISTRY as learner_registry


class MIR3Runner:
    def __init__(self, args, envs, eval_envs):
        self.args = args
        self.envs = envs
        self.eval_envs = eval_envs

        self.total_timesteps = args.total_timesteps
        self.buffer_size = args.buffer_size
        self.parallel_num = args.parallel_num
        self.episode_limit = args.episode_limit
        self.n_agents = args.n_env_agents

        # init log and model saving
        self.run_dir = args.run_dir
        self.log_dir = os.path.join(self.run_dir, "logs")
        if not self.args.evaluate:
            os.makedirs(self.log_dir, exist_ok=True)
        self.model_dir = os.path.join(self.run_dir, "models")
        if not self.args.evaluate:
            os.makedirs(self.model_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        print('initializing PMIC runner')
        print("Observation shape:", args.observation_shape)
        print("State shape:", args.state_shape)
        print("Action shape:", args.action_shape)
        print("Number of agents:", args.n_env_agents)

        # policy
        self.agents = agent_registry[args.victim_policy.agent_name](args.observation_shape, args.state_shape, args.action_shape, args.action_type, args.n_env_agents, device=args.device, **args.victim_policy.agent_args)
        self.learner = learner_registry[args.victim_policy.learner_name](self.agents, **args.victim_policy.learner_args)

        # buffer
        self.scheme = {
            "obs": {"vshape": (args.observation_shape,), "group": args.n_env_agents},
            "state": {"vshape": (args.state_shape,), "group": args.n_env_agents},
            "avail_actions": {"vshape": (args.action_shape,), "group": args.n_env_agents},
            "actions": {"vshape": (args.n_actions,), "group": args.n_env_agents},
            "actions_onehot": {"vshape": (args.action_shape,), "group": args.n_env_agents},
            "rewards": {"vshape": (1,), "group": args.n_env_agents},
            "env_rewards": {"vshape": (1,), "group": args.n_env_agents},
            "masks": {"vshape": (1,), "group": 1},
            # "actor_hs": {"vshape": (args.victim_policy.agent_args.hidden_dim,), "group": args.n_env_agents},
        }
        self.buffer = EpisodeBuffer(
            self.scheme,
            self.buffer_size,
            self.episode_limit + 1
        )
        
        self.window_size = args.pmic_window_size
        self.window = []
        self.pointer = 0
        self.mean_reward = -1000
        
        self.mi_buffer = MIBuffer(0, args.neg_buffer_size, args.n_env_agents, args.state_shape, args.action_shape)

        # load model
        if args.victim_policy.checkpoint_path:
            print("Load checkpoints from", args.victim_policy.checkpoint_path)
            self.agents.load(args.victim_policy.checkpoint_path)

    def slide(self, rewards):
        for reward in rewards:
            if len(self.window) == self.window_size:
                self.mean_reward = (self.mean_reward * self.window_size - self.window[self.pointer] + reward) / self.window_size
                self.window[self.pointer] = reward
                self.pointer = (self.pointer + 1) % self.window_size
            else:
                self.mean_reward = (self.mean_reward * len(self.window) + reward) / (len(self.window) + 1)
                self.window.append(reward)
                self.pointer = (self.pointer + 1) % self.window_size
            
    def reset(self):
        self.batch = EpisodeBatch(self.scheme, self.parallel_num, self.episode_limit + 1)
        self.envs.reset()
        obs = self.envs.get_obs()
        state = self.envs.get_state()
        avail_actions = self.envs.get_avail_actions()

        self.batch.add({"obs": obs, "state": state, "avail_actions": avail_actions, "masks": torch.ones((self.parallel_num,))}, t=0)

    def build_rewards(self):
        self.batch._meta["rewards"] = self.batch["env_rewards"]

    def run(self, start_t=0):
        if self.args.evaluate:
            self.eval(start_t)
            if self.args.save_replay:
                self.eval_envs.save_replay()
            return

        start = time.time()
        episode_time = []
        
        last_log = start_t
        last_save = start_t
        last_eval = start_t
        global_t = start_t
        episodes = 0

        self.eval(start_t)

        while global_t < self.total_timesteps:
            time_1 = time.time()
            self.reset()
            env_masks = np.ones((self.parallel_num,))
            # Collect
            self.agents.prep_rollout()
            actor_hs = self.agents.init_hidden(self.parallel_num)
            for step in range(self.episode_limit):
                # sample actions
                with torch.no_grad():
                    actions, actor_hs = self.learner.collect(self.batch, step, global_t, actor_hs=actor_hs)

                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.envs.step(cpu_actions)

                masks = 1 - dones
                env_masks = masks * env_masks

                obs = self.envs.get_obs()
                state = self.envs.get_state()
                avail_actions = self.envs.get_avail_actions()

                self.batch.add({"env_rewards": rewards}, step)
                self.batch.add({"obs": obs, "state": state, "avail_actions": avail_actions, "masks": env_masks,}, step + 1)

                if np.sum(env_masks) == 0:
                    break
            # get last actions and values (no need to perform a step)
            with torch.no_grad():
                actions, actor_hs = self.learner.collect(self.batch, step + 1, global_t, actor_hs=actor_hs)

            self.build_rewards()
            self.buffer.insert(self.batch)
            
            episode_rewards = torch.sum(self.batch["env_rewards"], 1).mean(1).squeeze(1) # [n_env, ]
            for i in range(self.parallel_num):
                for j in range(self.episode_limit + 1):
                    self.mi_buffer.add_neg(self.buffer["state"][i, j], self.buffer["actions_onehot"][i, j], 
                                                 episode_rewards[i])
            
            # learn
            train_info = {}
            for _ in range(self.args.train_epochs):
                # train club module
                club_loss = 0.0
                mine_loss = 0.0
                
                # consider buffer size and frequency
                if global_t - start_t >= 100:
                    club_loss = self.learner.learn_club(self.mi_buffer)
                    mine_loss = self.learner.learn_mine(self.mi_buffer)

                    if self.args.sample_timestep:
                        if self.buffer.can_sample_timestep(self.args.batch_size, self.args.num_batches):
                            self.agents.prep_training()
                            samples = self.buffer.sample_timestep(self.args.batch_size, self.args.num_batches)
                            train_info = self.learner.learn_with_pmic(samples, episodes)
                    else:
                        if self.buffer.can_sample(self.args.batch_size, self.args.num_batches):
                            self.agents.prep_training()
                            samples = self.buffer.sample(self.args.batch_size, self.args.num_batches)
                            train_info = self.learner.learn_with_pmic(samples, episodes)
                else:
                    if self.args.sample_timestep:
                        if self.buffer.can_sample_timestep(self.args.batch_size, self.args.num_batches):
                            self.agents.prep_training()
                            samples = self.buffer.sample_timestep(self.args.batch_size, self.args.num_batches)
                            train_info = self.learner.learn(samples, episodes)
                    else:
                        if self.buffer.can_sample(self.args.batch_size, self.args.num_batches):
                            self.agents.prep_training()
                            samples = self.buffer.sample(self.args.batch_size, self.args.num_batches)
                            train_info = self.learner.learn(samples, episodes)

            time_2 = time.time()
            episode_time.append(time_2 - time_1)
            if global_t - start_t > 5e4 and self.args.measure_time:
                episode_time = np.array(episode_time)
                time_mean = np.mean(episode_time)
                time_std = np.std(episode_time)
                with open('time.txt', 'a') as f:
                    f.write('Alg {}, Env {}, mean {}, std: {}\n'.format(self.args.victim_policy.learner_name
                            , self.args.env_args.map_name, time_mean, time_std))
                return

                
            # post_process
            global_t += (step + 1) * self.parallel_num
            episodes += self.parallel_num
            if global_t - last_save >= self.args.save_interval or global_t >= self.total_timesteps:
                self.agents.save(os.path.join(self.model_dir, str(global_t)))
                last_save = global_t

            # log information
            if global_t - last_log >= self.args.log_interval:
                end = time.time()
                print("Collect at timestep {}/{}, FPS {}".format(global_t, self.total_timesteps, int((global_t - start_t) / (end - start))))
                self.writer.add_scalar("train/mean_step_rewards", self.buffer["rewards"][:-1].mean().item(), global_t)
                self.writer.add_scalar("train/mean_step_env_rewards", self.buffer["env_rewards"][:-1].mean().item(), global_t)
                
                self.writer.add_scalar("train/club_loss", club_loss, global_t)
                self.writer.add_scalar("train/mine_loss", mine_loss, global_t)
                
                for k in train_info:
                    self.writer.add_scalar("train/" + k, train_info[k], global_t)
                last_log = global_t

            # eval
            if global_t - last_eval >= self.args.eval_interval or global_t >= self.total_timesteps:
                self.eval(global_t)
                last_eval = global_t

    def eval(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": [], "mi": [], "return_total": []}
            for step in range(self.args.episode_limit):
                with torch.no_grad():
                    actions, actor_hs = self.agents.perform(obs, actor_hs, avail_actions)
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

        nni.report_intermediate_result(np.mean([np.sum(elm) for elm in log_info["return"]]))
