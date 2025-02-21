#!/usr/bin/env python
import sys
import os
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config1 import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from runner.tacview import Tacview
import warnings

from datetime import date
warnings.filterwarnings("ignore", category=UserWarning)

import torch.utils.tensorboard as tb


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        if all_args.n_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                logging.error("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        if all_args.n_eval_rollout_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        if all_args.n_eval_rollout_threads == 1:
            return DummyVecEnv([get_env_fn(0)])
        else:
            return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    group = parser.add_argument_group("JSBSim Env parameters")
    group.add_argument('--scenario-name', type=str, default='singlecombat_simple',
                       help="Which scenario to run on")
    group.add_argument('--render-mode', type=str, default='txt',
                       help="txt or real_time")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def input_args(all_args,algorithm_name="dsac"):
    all_args.env_name = "SingleCombat"
    all_args.algorithm_name = algorithm_name
    all_args.scenario_name = "1v1/ShootMissile/Selfplay"
    # all_args.scenario_name = "1v1/ShootMissile/VsBaseline_nolimitSelfpaly"
    all_args.experiment_name = "v1"

    all_args.seed = 1
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 20
    all_args.cuda = True
    all_args.log_interval=1
    all_args.save_interval = 1

    all_args.use_eval = True
    all_args.n_eval_rollout_threads = 1
    all_args.eval_interval = 1
    all_args.eval_episodes = 1

    all_args.num_mini_batch = 5
    all_args.buffer_size = 1500
    all_args.num_env_steps = 3e9
    all_args.max_grad_norm = 0.5

    all_args.lr = 3e-4
    # all_args.lr =1e-4
    all_args.gamma = 0.99
    all_args.max_grad_norm = 2
    all_args.entropy_coef=1e-3
    if algorithm_name=="ppo":
        all_args.ppo_epoch = 4
        all_args.clip_param = 0.2

    all_args.hidden_size = "128 128"
    all_args.act_hidden_size = "128 128"
    all_args.recurrent_hidden_size = 128
    all_args.recurrent_hidden_layers = 1
    all_args.data_chunk_length = 8

    all_args.use_selfplay=True
    all_args.use_prior=True
    all_args.selfplay_algorithm="fsp"

    # 
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args = input_args(all_args,"ppo")
    # all_args = input_args_ppo(all_args)

    # all_args.model_dir="/home/sdj/home/sdj/graduation/final/LAG-1/scripts/results/SingleCombat/1v1/ShootMissile/Selfplay/ppo/v1/2025-02-19"

    all_args.use_wandb=False

    # seed
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        logging.info("choose to use gpu...")
        device = torch.device("cuda:0")  # use cude mask to control using which GPU
        torch.set_num_threads(all_args.n_training_threads)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        logging.info("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    today = date.today().strftime("%Y-%m-%d")
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name / today
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    writer = tb.SummaryWriter(run_dir)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    render_mode = all_args.render_mode
    
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "render_mode": render_mode,

        "writer": writer
    }

    # run experiments
    if all_args.env_name == "MultipleCombat":
        runner = ShareJSBSimRunner(config)
    else:
        if all_args.use_selfplay:
            from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner
        else:
            # 实际运行在这里调用了
            from runner.jsbsim_runner import JSBSimRunner as Runner
        runner = Runner(config)
    try:
        # 开始训练
        runner.run()
    except BaseException:
        traceback.print_exc()
    finally:
        # post process
        envs.close()

        # if all_args.use_wandb:
        #     run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
