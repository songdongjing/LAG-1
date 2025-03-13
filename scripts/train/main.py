#!/usr/bin/env python
import sys
import os
import traceback

import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from config import get_config

import warnings

from datetime import date
warnings.filterwarnings("ignore", category=UserWarning)

from scripts.train.parameter import input_args,parse_args,make_train_env,make_eval_env

import torch.utils.tensorboard as tb


def main(args):
    #算法参数设置，在parameter.py中设置
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args = input_args(all_args,"ppo",vsbaseline=True,render_mode="real_time")
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
    name="加上导弹惩罚"  #训练文件名字
    run_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results") \
        / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name / name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    writer = tb.SummaryWriter(run_dir)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name)
                              + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
        "render_mode": all_args.render_mode,

        "writer": writer
    }

    # run experiments
    if all_args.use_selfplay:
        from runner.selfplay_jsbsim_runner import SelfplayJSBSimRunner as Runner # 实际运行在这里调用了
    else:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:])
