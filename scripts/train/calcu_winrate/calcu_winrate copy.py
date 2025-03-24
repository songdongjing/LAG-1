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




ego_model_dir = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/train/calcu_winrate/model/ppo_entropy_20thread"
enm_model_dir = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/train/calcu_winrate/model/ppo_entropy_20thread"
baseline_model_dir = "/home/sdj/home/sdj/graduation/final/LAG-1/scripts/results/SingleCombat/1v1/ShootMissile/Selfplay/ppo/v1/MEppo+mlp+20thread"
class Args:
    def __init__(self) -> None:
        # 初始化方法，用于设置对象的初始状态
        # 设置增益参数，通常用于控制学习率或步长
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

class Args2:
    def __init__(self) -> None:
        # 初始化方法，用于设置对象的初始状态
        # 设置增益参数，通常用于控制学习率或步长
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
        self.use_recurrent_policy = False
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

def env_parallel(ego_model, enm_model, seed):

    ego_configs = {
        "camp": "red",
        "config": {
            "user_id": "admin",
            "password": "123456",
            "simulation_id": "uav_v1.0",
            "ip_address": None,
            "suffix": "_a1",
            "timeout": 0.5,
            "scenario_name": "SingleCombat/SingleCombat2.xml"
        },
        "action": {
            "discrete": True,
            "action_range": False,
            "rescale": False
        },
        "reward": {
            "step": {
                "goal_achieved": 1,
                "goal_failed": 0,
                "delt_time": 0
            },
            "end": {
                "win": 100,
                "loss": -10000,
                "accident": -10000
            },
            "time_limit": 1500,
            "sleep_time": 0
        },
        "seed": seed,
        "random": True,
        "num_missiles": 1
    }

    env = SingleCombatEnv(ego_configs)
    obs_shape = env.observation_space.shape
    num_agents = env.num_agents
    device = torch.device("cpu")
    act_space = env.action_space
    ego_args = Args()
    enm_args = Args()
    ego_policy = PPOActor(ego_args, obs_shape, act_space, device)
    enm_policy = PPOActor(enm_args, obs_shape, act_space, device)
    ego_policy.load_state_dict(torch.load(ego_model))
    # 忽略传入的enm_model，固定使用dodge_missile_model.pt
    dodge_missile_model = "dodge_missile_model.pt"
    try:
        enm_policy.load_state_dict(torch.load(dodge_missile_model))
        print(f"成功加载敌方模型: {dodge_missile_model}")
    except Exception as e:
        print(f"无法加载dodge_missile_model.pt: {e}，使用传入的模型: {enm_model}")
        enm_policy.load_state_dict(torch.load(enm_model))

    episode_opp_rewards = 0
    episode_ego_rewards = 0

    obs = env.reset()

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
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        # Obser reward and next obs
        obs, rewards, dones, infos = env.step(actions)
        rewards_ego = rewards[:num_agents // 2, ...] #这里把reward维度转了
        rewards_opp = rewards[num_agents // 2:, ...]

        episode_opp_rewards += rewards_opp
        episode_ego_rewards += rewards_ego
        if dones.all():
            break
        enm_obs =  obs[num_agents // 2:, ...]
        ego_obs =  obs[:num_agents // 2, ...]

    if episode_ego_rewards>=episode_opp_rewards:
        return 1
    else:
        return 0

def caculate_task(ego_agent, enm_agent):
    num_cores = os.cpu_count()#获取
    pool = Pool(num_cores)
    # pool = Pool(2)
    multi_res=[]
    count=15 #计算次数
    # for ego_agent in range(5,15):
    #     for enm_agent in range(0,ego_agent):
    #         multi_res.append(pool.apply_async(caculate_task, (ego_agent,enm_agent)))
    for i in range(count):
        multi_res.append(pool.apply_async(env_parallel,(ego_agent, enm_agent,random.randint(0,1000000))))
    pool.close()  #关闭pool，使其不再接受新的任务
    pool.join()  #主进程阻塞
    result=[]
    for res in multi_res:
        result.append(res.get())
    #从result中查找1的个数
    win_count=result.count(1)
    winrate=win_count/count
    return winrate
    # # 将result写入一个文件中
    # with open('result.txt', 'w') as f:
    #     for i in result:
    #         f.write(str(i))
    #         f.write('\n')


def main():
    winrate_list=[]
    count=100
    for i in range(count):
        ego_model=ego_model_dir+f"/actor_"+str(i)+".pt"
        # enm_model=ego_model_dir+f"/actor_"+str(j+100)+".pt"
        enm_model=baseline_model_dir+f"/actor_100.pt"
        winrate_list.append([i,caculate_task(ego_model,enm_model)])

    #将winrate_list写入一个文件
    with open('winrate_list.txt', 'w') as f:
        for i in winrate_list:
            f.write(str(i))
            f.write('\n')
    # data=data.transpose()#
    # 将字典转换为 DataFrame

    # df = pd.DataFrame(winrate_list,columns=['episode','ego_reward', 'opo_reward', 'win_rate'])

    # # 将 DataFrame 写入 CSV 文件
    # #文件存储位置修改
    
    # data=data.transpose()

    
    # # 将字典转换为 DataFrame
    # df = pd.DataFrame(data,columns=['episode','ego_reward', 'opo_reward', 'win_rate'])
    
    # df.to_csv(f'/testdata/有武器{ego_agent}VS{enm_agent}数据记录.csv', index=False)
# except:
#     print("发生错误")

if __name__ == '__main__':
    main()
    # 提取 x, y 坐标和对应的值
    # 数据列表，每个元素为 ((x, y), value)


# if __name__ == '__main__':
#     # main()
#     result=[[(5, 0), 0.0], [(5, 1), 0.39], [(5, 2), 0.88], [(5, 3), 0.88], [(5, 4), 0.36], [(6, 0), 0.0], [(6, 1), 0.33], [(6, 2), 0.88], [(6, 3), 0.88], [(6, 4), 0.07], [(6, 5), 0.1], [(7, 0), 0.0], [(7, 1), 0.8], [(7, 2), 0.88], [(7, 3), 0.88], [(7, 4), 0.99], [(7, 5), 0.99], [(7, 6), 1.0], [(8, 0), 0.0], [(8, 1), 0.34], [(8, 2), 0.88], [(8, 3), 0.88], [(8, 4), 0.17], [(8, 5), 0.16], [(8, 6), 0.41], [(8, 7), 0.03], [(9, 0), 0.0], [(9, 1), 0.33], [(9, 2), 0.91], [(9, 3), 0.9], [(9, 4), 0.18], [(9, 5), 0.17], [(9, 6), 0.51], [(9, 7), 0.01], [(9, 8), 0.5], [(10, 0), 0.0], [(10, 1), 0.47], [(10, 2), 0.9], [(10, 3), 0.9], [(10, 4), 0.14], [(10, 5), 0.17], [(10, 6), 0.48], [(10, 7), 0.01], [(10, 8), 0.44], [(10, 9), 0.43], [(11, 0), 0.0], [(11, 1), 0.5], [(11, 2), 0.9], [(11, 3), 0.9], [(11, 4), 0.15], [(11, 5), 0.15], [(11, 6), 0.49], [(11, 7), 0.0], [(11, 8), 0.47], [(11, 9), 0.46], [(11, 10), 0.49], [(12, 0), 0.0], [(12, 1), 0.31], [(12, 2), 0.9], [(12, 3), 0.9], [(12, 4), 0.07], [(12, 5), 0.1], [(12, 6), 0.28], [(12, 7), 0.0], [(12, 8), 0.28], [(12, 9), 0.27], [(12, 10), 0.29], [(12, 11), 0.31], [(13, 0), 0.0], [(13, 1), 0.3], [(13, 2), 0.9], [(13, 3), 0.9], [(13, 4), 0.07], [(13, 5), 0.11], [(13, 6), 0.27], [(13, 7), 0.0], [(13, 8), 0.26], [(13, 9), 0.26], [(13, 10), 0.27], [(13, 11), 0.29], [(13, 12), 0.28], [(14, 0), 0.0], [(14, 1), 0.32], [(14, 2), 0.9], [(14, 3), 0.89], [(14, 4), 0.14], [(14, 5), 0.14], [(14, 6), 0.49], [(14, 7), 0.0], [(14, 8), 0.44], [(14, 9), 0.45], [(14, 10), 0.47], [(14, 11), 0.47], [(14, 12), 0.48], [(14, 13), 0.46]]
#     #将result写入文件中
#     # with open('/home/sdj/home/sdj/newandgit/renders/result.txt', 'w') as f:
#     #     for i in result:
#     #         f.write(str(i))
#     #         f.write('\n')
#     #将result写入文件中


#     #将result列表中的元组转化为列表
#     result_list=[]
#     for i in result:
#         result_list.append([list(i[0])[0],list(i[0])[1],i[1]])
#     print(result_list)


    
#     df = pd.DataFrame(result_list,columns=['ego_agent','enm_agent','win_rate'])
#     df.to_csv('/home/sdj/home/sdj/newandgit/renders/result.csv', index=False)









