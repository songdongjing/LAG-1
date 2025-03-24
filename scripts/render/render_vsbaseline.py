import sys
sys.path.append('/home/sdj/home/sdj/newandgit/')
import random
import os
import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from envs.JSBSim.core.catalog import Catalog as c
from algorithms.ppo.ppo_actor_backup import PPOActor
import logging
from multiprocessing import Pool
logging.basicConfig(level=logging.DEBUG)
import pandas as pd
import os
# 添加路径
import sys
from gymnasium import spaces




ego_model_dir = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/train/data_collect/MEPPO+20thread+GRU/actor_latest.pt"
enm_model_dir = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/train/data_collect/MEPPO+20thread+GRU/actor_latest.pt"
class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        # 设置隐藏层大小，这里使用字符串表示两个隐藏层，每层大小为128
        self.hidden_size = '128 128'
        # 设置激活函数的隐藏层大小，与隐藏层大小相同
        self.act_hidden_size = '128 128'
        # 设置激活函数的ID，可能用于选择不同的激活函数
        self.activation_id = 1
        # 设置是否使用特征归一化，默认为False
        self.use_feature_normalization = False

        # 设置是否使用循环策略，默认为True
        self.use_recurrent_policy = True
        # 设置循环隐藏层的大小，默认为128
        self.recurrent_hidden_size = 128
        # 设置循环隐藏层的层数，默认为1
        self.recurrent_hidden_layers = 1
        # 设置张量的数据类型和设备，这里使用float32类型和CPU设备
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True

def _t2n(x):
    return x.detach().cpu().numpy()
#任务定义为

def fight_render(ego_model, enm_model):
    vsbaseline = True

    env = SingleCombatEnv("/home/sdj/home/sdj/graduation/final/LAG-1/envs/JSBSim/configs/1v1/ShootMissile/Selfplay_fardistance")
    num_agents = 2
    args = Args()
    #创建策略模型
    ego_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
    if vsbaseline:
        enm_policy = PPOActor(args, env.observation_space, spaces.MultiDiscrete([41, 41, 41, 30]), device=torch.device("cuda"))
    else:
        enm_policy = PPOActor(args, env.observation_space, env.action_space, device=torch.device("cuda"))
    #加载策略模型
    ego_policy.eval()
    enm_policy.eval()

    # 加载模型
    ego_policy.load_state_dict(torch.load(ego_model), strict=False)
    enm_policy.load_state_dict(torch.load(enm_model), strict=False)

    episode_opp_rewards=0
    episode_ego_rewards=0

    obs = env.reset()
    env.render(mode='txt', filepath=f'scripts/render/fight.txt.acmi')

    my_actions=np.random.randint(0, 10, size=(2,4))
    my_actions=my_actions.astype(np.float32)
    ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
    masks = np.ones((num_agents // 2, 1))
    enm_obs =  obs[num_agents // 2:, :]
    ego_obs =  obs[:num_agents // 2, :]
    enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)
    while True:
        ego_actions, _, ego_rnn_states = ego_policy(ego_obs, ego_rnn_states, masks, deterministic=True)
        ego_actions = _t2n(ego_actions)
        ego_rnn_states = _t2n(ego_rnn_states)
        enm_actions, _, enm_rnn_states = enm_policy(enm_obs, enm_rnn_states, masks, deterministic=True)

        enm_actions = _t2n(enm_actions)
        enm_rnn_states = _t2n(enm_rnn_states)

         #为了保持与ego_actions的维度一致
        if vsbaseline:
            enm_actions = np.append(enm_actions, [[0.]], axis=1).astype(np.float32)
        
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        obs, rewards, dones, infos = env.step(actions)
        rewards_ego = rewards[:num_agents // 2, ...] #这里把reward维度转了
        rewards_opp = rewards[num_agents // 2:, ...]

        episode_opp_rewards += rewards_opp
        episode_ego_rewards += rewards_ego

        env.render(mode='txt', filepath=f'scripts/render/fight.txt.acmi')
        if dones.all():
            break
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    if episode_ego_rewards>=episode_opp_rewards:
        return 1
    else:
        return 0


def main():
    flag=fight_render(ego_model_dir, enm_model_dir)
    print(f'The result is: {flag}')


if __name__ == '__main__':
    main()










