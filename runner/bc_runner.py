import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer
from .jsbsim_runner import JSBSimRunner

#############################
# 数据收集
#这里采集的数据是最新一代智能体和自己对抗的数据
#############################
def _t2n(x):
    return x.detach().cpu().numpy()


class bc_runner(JSBSimRunner):

    def load(self):
        self.use_selfplay = self.all_args.use_selfplay 
        assert self.use_selfplay == True, "Only selfplay can use SelfplayRunner"
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.num_opponents = self.all_args.n_choose_opponents
        assert self.eval_episodes >= self.num_opponents, \
        f"Number of evaluation episodes:{self.eval_episodes} should be greater than number of opponents:{self.num_opponents}"
        self.init_elo = self.all_args.init_elo
        self.latest_elo = self.init_elo

        # 创建并加载专家智能体模型
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_policy_backup import PPOPolicy as Policy
        elif self.algorithm_name == "dsac":
            from algorithms.dsac.dsac_policy import DSACPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        actor_path = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/train/data_collect/MEPPO+20thread+GRU/actor_latest.pt"
        policy_actor_state_dict = torch.load(actor_path, weights_only=True)
        self.policy.actor.load_state_dict(policy_actor_state_dict, strict=False)
        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents // 2, self.obs_space, self.act_space)

        # [Selfplay] allocate memory for opponent policy/data in training
        from algorithms.utils.selfplay import get_algorithm
        self.selfplay_algo = get_algorithm(self.all_args.selfplay_algorithm)

        assert self.num_opponents <= self.n_rollout_threads, \
            "Number of different opponents({}) must less than or equal to number of training threads({})!" \
            .format(self.num_opponents, self.n_rollout_threads)
        self.policy_pool = {}  # type: dict[str, float]
        self.opponent_policy = [Policy(self.all_args, self.obs_space, self.act_space, device=self.device)]
        # 让对手智能体也加载相同的预训练模型
        self.opponent_policy[0].actor.load_state_dict(policy_actor_state_dict, strict=False)
        
        self.opponent_env_split = np.array_split(np.arange(self.n_rollout_threads), len(self.opponent_policy))
        self.opponent_obs = np.zeros_like(self.buffer.obs[0])
        self.opponent_rnn_states = np.zeros_like(self.buffer.rnn_states_actor[0])
        self.opponent_masks = np.ones_like(self.buffer.masks[0])

        logging.info("\n Load selfplay opponents: Algo {}, num_opponents {}.\n"
                        .format(self.all_args.selfplay_algorithm, self.num_opponents))



    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        # [Selfplay] get actions of opponent policy
        opponent_actions = np.zeros_like(actions)
        for policy_idx, policy in enumerate(self.opponent_policy):
            env_idx = self.opponent_env_split[policy_idx]
            opponent_action, opponent_rnn_states \
                = policy.act(np.concatenate(self.opponent_obs[env_idx]),
                                np.concatenate(self.opponent_rnn_states[env_idx]),
                                np.concatenate(self.opponent_masks[env_idx]))
            opponent_actions[env_idx] = np.array(np.split(_t2n(opponent_action), len(env_idx)))
            self.opponent_rnn_states[env_idx] = np.array(np.split(_t2n(opponent_rnn_states), len(env_idx)))
        actions = np.concatenate((actions, opponent_actions), axis=1)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def collect_data(self, step):
        #通过自博弈生成大量数据并保存
        obs = self.envs.reset()
        for step in range(step):
            # Sample actions
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

            data=obs
            # Obser reward and next obs
            obs, rewards, dones, infos = self.envs.step(actions)

            data = [data, actions]

            # insert data into txt
            self.write_data(data)


    def write_data(self, data):
        with open("scripts/train/data_collect/data.txt", 'a') as f:
            f.write(str(data))
            f.write("\n")

