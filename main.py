import os
import yaml
import torch
import argparse
import collections
import setproctitle
import numpy as np
import nni
from gym import spaces 
from pprint import pprint
from easydict import EasyDict

from envs import REGISTRY as env_registry
from envs.vec_env import SubProcVecEnv, DummyEnv


def get_config(config_name, subfolder):
    with open(os.path.join("configs", subfolder, "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def make_train_envs(args):
    env_fn = env_registry[args.env]
    if args.parallel_num == 1:
        return DummyEnv(env_fn, 1, args.episode_limit, args.seed, args.env_args)
    else:
        return SubProcVecEnv(env_fn, args.parallel_num, args.episode_limit, args.seed, args.env_args)


def make_eval_envs(args):
    env_fn = env_registry[args.env]
    return DummyEnv(env_fn, 1, args.episode_limit, args.seed + 50000, args.eval_env_args)


if __name__ == "__main__":
    # parse args, get default and specific configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='sc2')
    parser.add_argument("--alg", type=str, default='maddpg_smac')
    parser.add_argument("--change_t", type=int, default=-1)
    parser.add_argument("--map", type=str, default='')
    parser.add_argument("--gpu", type=str, default='')
    parser.add_argument("--victim_checkpoint", type=str, default='')
    parser.add_argument("--adversary_checkpoint", type=str, default='')
    parser.add_argument("--token", type=str, default='train')
    parser.add_argument("--param_club", type=float, default=0.)
    parser.add_argument("--start_t", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adversarial", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--adv_training_evaluate_filename", type=str, default='evaluator.csv')
    parser.add_argument("--measure_time", action="store_true", default=False)
    parser.add_argument("--adv_agent_id", type=int, default=-1)
    parser.add_argument("--nr_agents", type=int, default=-1)
    parser.add_argument("--actor_lr", type=float, default=-1.)
    parser.add_argument("--critic_lr", type=float, default=-1.)
    args = parser.parse_args()

    config = get_config("default", "")
    env_config = get_config(args.env, "envs")
    alg_config = get_config(args.alg, "algs")
    start_t = args.start_t
    if args.map:
        env_config['env_args']['map_name'] = args.map
    if args.gpu:
        config['device'] = args.gpu
    if args.token:
        config['token'] = args.token
    if 'learner_args' in alg_config['victim_policy'] and 'param_club' in alg_config['victim_policy']['learner_args']:
        alg_config['victim_policy']['learner_args']['param_club'] = args.param_club
    config['evaluate_adversarial'] = args.adversarial
    if args.evaluate:
        config['evaluate'] = True
    config['adv_training_evaluate_filename'] = args.adv_training_evaluate_filename
    config['measure_time'] = args.measure_time
    if args.adv_agent_id != -1:
        alg_config['adversary_policy']['adv_agent_ids'] = [args.adv_agent_id]
    if args.nr_agents >= 1 and args.env == 'robot':
        env_config['env_args']['nr_agents'] = args.nr_agents
    if args.actor_lr > 0:
        alg_config['victim_policy']['learner_args']['actor_lr'] = args.actor_lr
    if args.critic_lr > 0:
        alg_config['victim_policy']['learner_args']['critic_lr'] = args.critic_lr        

    config = recursive_dict_update(config, env_config)
    config = recursive_dict_update(config, alg_config)
    if args.victim_checkpoint:
        config['victim_policy']['checkpoint_path'] = args.victim_checkpoint
    if args.adversary_checkpoint:
        config['adversary_policy']['checkpoint_path'] = args.adversary_checkpoint
    
    if args.seed:
        config['seed'] = args.seed

    if args.change_t >= 0:
        config["change_t"] = args.change_t
        print("Set change_t to", config["change_t"])

    # copy env args to eval env args
    if "eval_env_args" not in config:
        config["eval_env_args"] = config["env_args"]
    else:
        for key in config["env_args"]:
            if key not in config["eval_env_args"]:
                config["eval_env_args"][key] = config["env_args"][key]

    args = EasyDict(config)
    args.device = torch.device(args.device)

    # set random seed
    torch.set_num_threads(args.training_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # get basic information of env
    env = env_registry[args.env](**args.env_args)
    args.observation_shape = np.prod(env.observation_spaces[0].shape)
    args.state_shape = np.prod(env.state_space.shape)
    if isinstance(env.action_spaces[0], spaces.Discrete):
        args.action_shape = env.action_spaces[0].n
        args.n_actions = 1
        args.action_type = "discrete"
    elif isinstance(env.action_spaces[0], spaces.Box) or env.action_spaces[0].__class__.__name__ == "Box":
        args.action_shape = env.action_spaces[0].shape[0]
        args.n_actions = env.action_spaces[0].shape[0]
        args.action_type = "box"
    else:
        print(env.action_spaces[0].__class__.__name__)
        raise NotImplementedError

    args.episode_limit = env.episode_limit
    args.n_env_agents = env.n_agents
    args.adversary_policy.adv_agent_ids = [i % env.n_agents for i in args.adversary_policy.adv_agent_ids]
    env.close()

    # pprint(args)
    # make envs
    envs = make_train_envs(args)
    eval_envs = make_eval_envs(args)

    # create logging directory

    run_dir = os.path.join(
        "./results",
        args.env,
        args.env_args.map_name,
        args.adversarial,
        args.adversary_policy.learner_name if args.adversarial != "none" else args.victim_policy.learner_name,
        args.token,
        str(args.seed),
    )
    
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir
    setproctitle.setproctitle(run_dir)

    if not args.evaluate:
        with open(os.path.join(args.run_dir, "config.yaml"), "w") as fp:
            yaml.dump(dict(args), fp)
    # run experiments
    if args.adversarial == "traitor_ca_mi":
        print("Training adversarial traitors and caculating mutual information...")
        from runners.offpolicy.traitor_mi_runner import TraitorMIRunner as Runner
    elif args.adversarial == "none":
        if args.victim_policy.learner_name == "mir3":
            print("Training MIR3 policies...")
            from runners.offpolicy.mir3_runner import MIR3Runner as Runner
        elif args.victim_policy.learner_name == "romq":
            print("Training ROMQ policies...")
            from runners.offpolicy.romq_runner import ROMQRunner as Runner
        else:
            print("Training MADDPG/M3DDPG/ERNIE/ROMAX policies...")
            from runners.offpolicy.base_runner import BaseRunner as Runner
    else:
        raise NotImplementedError
    runner = Runner(args, envs, eval_envs)
    if start_t != 0:
        runner.run(start_t)
    else:
        runner.run()

    # post process
    envs.close()
    eval_envs.close()