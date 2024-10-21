import os
import nni
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import csv

from data.episode_buffer import EpisodeBatch, EpisodeBuffer
from agents import REGISTRY as agent_registry
from learners import REGISTRY as learner_registry


class BaseRunner:
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
        self.adv_agent_ids = np.array(args.adversary_policy.adv_agent_ids, dtype=np.int64)
        self.n_adv_agents = len(self.adv_agent_ids)
        self.adv_agents = agent_registry[args.adversary_policy.agent_name](args.observation_shape, args.state_shape, args.action_shape, args.action_type, self.n_adv_agents, device=args.device, **args.adversary_policy.agent_args)

        # buffer
        self.scheme = {
            "obs": {"vshape": (args.observation_shape,), "group": args.n_env_agents},
            "noise_obs": {"vshape": (args.observation_shape,), "group": args.n_env_agents},
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

        self.batch.add({"obs": obs, "state": state, "avail_actions": avail_actions, "masks": torch.ones((self.parallel_num,))}, t=0)

    def build_rewards(self):
        self.batch._meta["rewards"] = self.batch["env_rewards"]

    def run(self, start_t=0):
        if self.args.evaluate:
            if self.args.eval_type == "obs" or self.args.evaluate_adversarial == 1:
                self.eval_obs(start_t)
            elif self.args.eval_type == "action" or self.args.evaluate_adversarial == 2:
                self.eval_action(start_t)
            elif self.args.evaluate_adversarial == 3:
                self.eval_m3ddpg(start_t)
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

        self.eval(global_t)

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
            # print(step)
            # get last actions and values (no need to perform a step)
            self.learner.collect(self.batch, step + 1, global_t, actor_hs=actor_hs)

            self.build_rewards()
            self.buffer.insert(self.batch)
        
            # learn
            train_info = {}
            for train_step in range(self.args.train_epochs):
                # print('{} / {}'.format(train_step, self.args.train_epochs)) 
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
        draw_tSNE = False
        if draw_tSNE:
            representation_list = []
            distribution_list = []
            state_list = []
            choose_list = []
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            if draw_tSNE:
                episode_state_list = []
                episode_representation_list = []
                episode_distribution_list = []
                episode_choose_list = np.zeros([self.n_agents, self.args.action_shape])
                episode_state_list.append(self.eval_envs.get_state())
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": [], "return_total": []}
            for step in range(self.args.episode_limit):
                with torch.no_grad():
                    if draw_tSNE:
                        episode_representation_list.append(self.agents.perform_action_representation(obs, actor_hs))
                    actions, actor_hs, action_logits = self.agents.perform_logits(obs, actor_hs, avail_actions)
                    if draw_tSNE:
                        episode_distribution_list.append(action_logits)
                        for parallel in range(len(actions)):
                            for agent_id in range(self.n_agents):
                                episode_choose_list[agent_id][actions[parallel][agent_id]] += 1

                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                episode_info["return"].append(rewards[0][0])
                episode_info["return_total"].append(sum(rewards[0]))
                for key in infos[0]:
                    if key not in episode_info:
                        episode_info[key] = []
                    episode_info[key].append(infos[0][key])

                obs = self.eval_envs.get_obs()
                if draw_tSNE:
                    episode_state_list.append(self.eval_envs.get_state())
                avail_actions = self.eval_envs.get_avail_actions()

                if np.all(dones):
                    episode_info["ep_length"].append(step + 1)
                    break
                    
            for key in episode_info:
                if key not in log_info:
                    log_info[key] = []
                log_info[key].append(episode_info[key])
            if draw_tSNE:
                state_list.append(episode_state_list)
                representation_list.append(episode_representation_list)
                distribution_list.append(episode_distribution_list)
                choose_list.append(episode_choose_list)

        if draw_tSNE:
            min_length = 1000
            for state in state_list:
                if len(state) < min_length:
                    min_length = len(state)
            for i in range(len(state_list)):
                state_list[i] = state_list[i][:min_length]
                representation_list[i] = torch.stack(representation_list[i][:min_length-1]).detach().cpu().numpy()
                distribution_list[i] = torch.stack(distribution_list[i][:min_length-1]).detach().cpu().numpy()
            test_token = 'maddpg'
            np.save('./club_defense/action_representation/{}.npy'.format(test_token), np.array(representation_list))
            np.save('./club_defense/action_distribution/{}.npy'.format(test_token), np.array(distribution_list))
            np.save('./club_defense/action_distribution/{}_state.npy'.format(test_token), np.array(state_list))
            np.save('./club_defense/action_choose/{}.npy'.format(test_token), np.array(choose_list))

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

    def eval_obs(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            adv_actor_hs = self.adv_agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            state = self.eval_envs.get_state()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": []}
            adversary = torch.randint(self.agents.n_agents, (1,)).item()
            for step in range(self.args.episode_limit):
                # print("{}/{}, {}/{}".format(step, self.args.episode_limit, episode, self.args.eval_episodes))
                perturb_obs = self.agents.attack_obs(obs, actor_hs, avail_actions, state, self.adv_agent_ids, self.args.eval_iter, self.args.eval_epsilon, self.args.eval_alpha, self.adv_agents.actor, adv_actor_hs)
                with torch.no_grad():
                    _, adv_actor_hs = self.adv_agents.perform(obs[:, self.adv_agent_ids], adv_actor_hs, avail_actions[:, self.adv_agent_ids])
                    actions, actor_hs = self.agents.perform(perturb_obs, actor_hs, avail_actions)

                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                episode_info["return"].append(rewards[0][0])
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

        nni.report_final_result(np.mean([np.sum(elm) for elm in log_info["return"]]))

    def eval_action(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            state = self.eval_envs.get_state()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": []}
            adversary = torch.randint(self.agents.n_agents, (1,)).item()
            for step in range(self.args.episode_limit):
                # print("{}/{}, {}/{}".format(step, self.args.episode_limit, episode, self.args.eval_episodes))
                with torch.no_grad():
                    actions, actor_hs, actions_logits = self.agents.perform_logits(obs, actor_hs, avail_actions)
                actions = self.agents.attack_action(actions, actions_logits, avail_actions, state, adversary, self.args.eval_iter, self.args.eval_epsilon, self.args.eval_alpha)

                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                episode_info["return"].append(rewards[0][0])
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

        nni.report_final_result(np.mean([np.sum(elm) for elm in log_info["return"]]))

    def eval_minq(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        obs_traj = []
        state_traj = []
        actions_traj = []
        logits_traj = []
        rewards_traj = []

        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            state = self.eval_envs.get_state()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": []}
            adversary = 0
            if self.args.change_t == -1:
                adversary = torch.randint(self.agents.n_agents, (1,)).item()
            for step in range(self.args.episode_limit):
                with torch.no_grad():
                    actions, actor_hs, action_logits = self.agents.perform_logits(obs, actor_hs, avail_actions)
                if step == self.args.change_t or self.args.change_t == -1:
                    actions, q_values = self.agents.attack_minq(actions, avail_actions, state, adversary)
                    # np.save("../traj/q_values_{}.npy".format(self.args.change_t), q_values)

                cpu_actions = actions.cpu().numpy()

                obs_traj.append(obs)
                state_traj.append(state)
                actions_traj.append(cpu_actions)
                logits_traj.append(action_logits.cpu().numpy())

                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                rewards_traj.append(rewards[0][0])

                episode_info["return"].append(rewards[0][0])
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

        obs_traj = np.concatenate(obs_traj, axis=0)
        state_traj = np.concatenate(state_traj, axis=0)
        actions_traj = np.concatenate(actions_traj, axis=0)
        logits_traj = np.concatenate(logits_traj, axis=0)
        rewards_traj = np.array(rewards_traj)

        # np.save("../traj/obs_{}.npy".format(self.args.change_t), obs_traj)
        # np.save("../traj/state_{}.npy".format(self.args.change_t), state_traj)
        # np.save("../traj/actions_{}.npy".format(self.args.change_t), actions_traj)
        # np.save("../traj/logits_{}.npy".format(self.args.change_t), logits_traj)
        # np.save("../traj/rewards_{}.npy".format(self.args.change_t), rewards_traj)

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

        nni.report_final_result(np.mean([np.sum(elm) for elm in log_info["return"]]))

    def eval_m3ddpg(self, total_t):
        log_info = {}
        self.agents.prep_rollout()
        # this reset is really really strange in SMAC environment!!!!!
        self.eval_envs.reset()
        for episode in range(self.args.eval_episodes):
            # self.eval_envs.reset()
            actor_hs = self.agents.init_hidden(1)
            obs = self.eval_envs.get_obs()
            state = self.eval_envs.get_state()
            avail_actions = self.eval_envs.get_avail_actions()
            episode_info = {"return": [], "ep_length": []}
            for step in range(self.args.episode_limit):
                # print("{}/{}, {}/{}".format(step, self.args.episode_limit, episode, self.args.eval_episodes))
                with torch.no_grad():
                    actions, actor_hs, actions_logits = self.agents.perform_logits(obs, actor_hs, avail_actions)
                # adversary = torch.randint(self.agents.n_agents, (1,)).item()
                actions = self.agents.attack_m3ddpg(actions_logits, avail_actions, state, self.args.eval_iter, self.args.eval_epsilon, self.args.eval_alpha, adversary=None)
                cpu_actions = actions.cpu().numpy()
                rewards, dones, infos = self.eval_envs.step(cpu_actions)

                episode_info["return"].append(rewards[0][0])
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
        with open(self.args.adv_training_evaluate_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            data = [score, total_t, 0.0, score, score]
            writer.writerow(data)
