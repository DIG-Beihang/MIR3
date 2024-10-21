import os
import time
import torch
import csv
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data.episode_buffer import EpisodeBatch, EpisodeBuffer
import nni
from agents import REGISTRY as agent_registry
from learners import REGISTRY as learner_registry


class ROMQRunner:
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

        print("Observation shape:", args.observation_shape)
        print("State shape:", args.state_shape)
        print("Action shape:", args.action_shape)
        print("Number of agents:", args.n_env_agents)

        # policy
        self.agents = agent_registry[args.victim_policy.agent_name](args.observation_shape, args.state_shape, args.action_shape, args.action_type, args.n_env_agents, device=args.device, **args.victim_policy.agent_args)
        self.learner = learner_registry[args.victim_policy.learner_name](self.agents, **args.victim_policy.learner_args)

        # buffer
        self.scheme = {
            "obs": {"vshape": (args.observation_shape,), "group": args.n_env_agents, "process": ["t-1"]},
            "noise_obs": {"vshape": (args.observation_shape,), "group": args.n_env_agents},
            "state": {"vshape": (args.state_shape,), "group": args.n_env_agents},
            "avail_actions": {"vshape": (args.action_shape,), "group": args.n_env_agents},
            "actions": {"vshape": (args.n_actions,), "group": args.n_env_agents},
            "actions_onehot": {"vshape": (args.action_shape,), "group": args.n_env_agents},
            "cf_actions_onehot": {"vshape": (args.action_shape,), "group": args.n_env_agents, "process": ["t-1"]},
            "rewards": {"vshape": (1,), "group": args.n_env_agents},
            "env_rewards": {"vshape": (1,), "group": args.n_env_agents},
            "masks": {"vshape": (1,), "group": 1},
            "episode_adversary": {"vshape": (1,), "group": 1},
            "agent_adversary": {"vshape": (1,), "group": 1},
            "noise_type": {"vshape": (1,), "group": args.n_env_agents},
        }
        self.buffer = EpisodeBuffer(
            self.scheme,
            self.buffer_size,
            self.episode_limit + 1
        )

        # load model
        if args.victim_policy.checkpoint_path:
            print("Load checkpoints from", args.victim_policy.checkpoint_path)
            self.agents.load(args.victim_policy.checkpoint_path)

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

        self.batch.add({"obs": obs, "state": state, "avail_actions": avail_actions, "masks": torch.ones((self.parallel_num,))}, t=0)

    def build_rewards(self):
        self.batch._meta["rewards"] = self.batch["env_rewards"]

    def run(self, start_t=0):
        if self.args.evaluate:
            if (self.args.eval_type == "action" or self.args.evaluate_adversarial == 2):
                self.eval_adv_training(start_t)
            else:
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
                actions, actor_hs = self.learner.collect(self.batch, step, global_t, actor_hs=actor_hs)

                cpu_actions = actions.detach().cpu().numpy()
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
            self.learner.collect(self.batch, step + 1, global_t, actor_hs=actor_hs)

            self.build_rewards()
            self.buffer.insert(self.batch)
        
            # learn
            train_info = {}
            for _ in range(self.args.train_epochs): 
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
            global_t += step + 1
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
            episode_info = {"return": [], "ep_length": [], "return_total": []}
            for step in range(self.args.episode_limit):
                # print(episode, step)
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

    def eval_adv_training(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        action_type = self.agents.action_type
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            state = self.eval_envs.get_state()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": [], "return_total": []}
            agent_adversary = torch.randint(self.agents.n_agents, (1,)).item()
            for step in range(self.args.episode_limit):
                # print(episode, step)
                obs = self.agents.check(obs)
                state = self.agents.check(state)
                if avail_actions is not None:
                    avail_actions = self.agents.check(avail_actions)
                if actor_hs is not None:
                    actor_hs = self.agents.check(actor_hs)
                actions_ori, actor_hs = self.agents.actor(obs, actor_hs, avail_actions)
                if action_type == 'discrete' and total_t > 1000000:

                    actions_shape = avail_actions.shape[-1]
                    
                    q_values, _ = self.agents.minq_net(obs[:, agent_adversary], None)
                    q_values[avail_actions[:, agent_adversary] == 0] = 1e10

                    actions = F.one_hot(q_values.argmin(dim=-1).detach(), num_classes=actions_shape) # [N, actions_shape]
                    actions[actions==0] = -1e10

                    actions_ori = actions_ori.detach().clone()
                    actions_ori[:, agent_adversary] = actions
                    actions_onehot = F.gumbel_softmax(actions_ori, hard=True)
                    actions = actions_onehot.argmax(dim=-1)
                elif action_type == 'box' and total_t > 1000000:
                    actions_onehot_ori = torch.tanh(actions_ori)
                    actions_onehot = actions_onehot_ori.detach().clone()
                    for _ in range(self.learner.action_iter):
                        actions_onehot.requires_grad_(True)
                        q_value, _ = self.agents.critic(state, actions_onehot, None)
                        q_value.sum().backward()
                        actions_onehot_grad = actions_onehot.grad.sign()[:, agent_adversary]
                        actions_onehot = actions_onehot.detach()
                        actions_onehot[:, agent_adversary] -= self.learner.action_alpha * actions_onehot_grad
                        actions_onehot = (torch.clamp(actions_onehot - actions_onehot_ori, -self.learner.action_epsilon, self.learner.action_epsilon) + actions_onehot_ori).detach()
                    actions = actions_onehot
                else:
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
        
        if "reward_run" in log_info.keys():
            score = np.mean([np.sum(elm) for elm in log_info["reward_run"]])
        else:
            score = np.mean([np.sum(elm) for elm in log_info["return"]])
        nni.report_intermediate_result(score) 
        with open(self.args.adv_training_evaluate_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            data = [score, total_t, 0.0, score, score]
            writer.writerow(data)