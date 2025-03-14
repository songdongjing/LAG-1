from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runner.tacview import Tacview
import wandb
import socket
import random
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            else:
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
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
            else:
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 1000)
            return env
        return init_env
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

#算法参数配置

#所有的参数可以在config.py中设置
def input_args(all_args,algorithm_name="ppo",vsbaseline=False,render_mode="txt"):
    all_args.env_name = "SingleCombat" #1V1空战环境 
    all_args.algorithm_name = algorithm_name  #算法名称
    all_args.render_mode = render_mode #渲染模式
    all_args.eval_interval = 1
    if vsbaseline:
        all_args.scenario_name = "1v1/ShootMissile/VsBaseline_nolimitSelfpaly"
        all_args.use_selfplay=False
    else:
        all_args.scenario_name = "1v1/ShootMissile/Selfplay_fardistance" #环境名称
        all_args.use_selfplay=False
        all_args.selfplay_algorithm="pfsp" #优先级自博弈
    all_args.experiment_name = "学习率1e-4" #实验名称
    all_args.use_prior=True #alpha-beta分布先验

    all_args.seed = random.randint(0, 1000000)
    all_args.n_training_threads = 1
    all_args.n_rollout_threads = 1
    all_args.cuda = True
    all_args.log_interval=1
    all_args.save_interval = 1

    all_args.use_eval = True
    all_args.n_eval_rollout_threads = 1
    all_args.eval_interval = 1
    all_args.eval_episodes = 1

    all_args.num_mini_batch = 5
    all_args.buffer_size = 1500
    all_args.num_env_steps = 5e5
    all_args.max_grad_norm = 0.5

    all_args.lr =1e-4
    all_args.gamma = 0.99
    all_args.max_grad_norm = 2
    all_args.entropy_coef=1e-3
    if algorithm_name=="ppo":
        all_args.ppo_epoch = 4
        all_args.clip_param = 0.2

    all_args.hidden_size = "128 128"
    all_args.act_hidden_size = "128 128"

    #使用GRU
    all_args.use_recurrent_policy=True
    all_args.recurrent_hidden_size = 128
    all_args.recurrent_hidden_layers = 1
    all_args.data_chunk_length = 8

    # 
    return all_args