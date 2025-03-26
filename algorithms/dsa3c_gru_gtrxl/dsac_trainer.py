import torch
import torch.nn as nn
from typing import Union, List
from .dsac_policy import DSACPolicy
from .dsac_critic import DSACCritic
from ..utils.buffer import ReplayBuffer
from ..utils.utils import check, get_gard_norm
import torch.nn.functional as F
import numpy as np

class DSACTrainer():
    def __init__(self, args, device=torch.device("cpu")):

        self.use_transformer = args.use_transformer
        self.use_recurrent_policy = args.use_recurrent_policy
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = -1
        self.gamma = 0.95
        self.tau = 0.005
        # rnn configs
        self.data_chunk_length = args.data_chunk_length

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def dsac_update(self, policy: DSACPolicy, sample):

        obs_batch, actions_batch, masks_batch, old_action_log_probs_batch, advantages_batch, \
            returns_batch, value_preds_batch, rnn_states_actor_batch, rnn_states_critic_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        advantages_batch = check(advantages_batch).to(**self.tpdv)
        returns_batch = check(returns_batch).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)

        critic_1_q_values, _ = policy.critic1(obs_batch, rnn_states_critic_batch, masks_batch)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, value_preds_batch.detach()))
        critic_2_q_values, _ = policy.critic2(obs_batch, rnn_states_critic_batch, masks_batch)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, value_preds_batch.detach()))
        critic_3_q_values, _ = policy.critic3(obs_batch, rnn_states_critic_batch, masks_batch)
        critic_3_loss = torch.mean(
            F.mse_loss(critic_3_q_values, value_preds_batch.detach()))

        policy.optimizer_critic1.zero_grad()
        critic_1_loss.backward()
        policy.optimizer_critic1.step()

        policy.optimizer_critic2.zero_grad()
        critic_2_loss.backward()
        policy.optimizer_critic2.step()

        policy.optimizer_critic3.zero_grad()
        critic_3_loss.backward()
        policy.optimizer_critic3.step()

        values, action_log_probs, _ = policy.evaluate_actions(obs_batch,
                                                             rnn_states_actor_batch,
                                                             rnn_states_critic_batch,
                                                             actions_batch,
                                                             masks_batch)
        entropy = -torch.sum(action_log_probs.exp() * (action_log_probs), dim=1, keepdim=True)
        # DSAC中使用两个Q网络的最小值来计算Actor损失
        q1_value = policy.critic1(obs_batch, rnn_states_critic_batch, masks_batch)
        q2_value = policy.critic2(obs_batch, rnn_states_critic_batch, masks_batch)
        q3_value = policy.critic3(obs_batch, rnn_states_critic_batch, masks_batch)
        mean_two_smallest = policy.average_of_two_smallest(q1_value[0], q2_value[0], q3_value[0])
        min_qvalue = torch.sum(action_log_probs.exp() * mean_two_smallest, dim=1, keepdim=True)
        #min_qvalue = torch.sum(torch.tensor(action_log_probs.exp()) * mean_two_smallest,dim=1,keepdim=True)

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        policy.optimizer_actor.zero_grad()
        actor_loss.backward()
        policy.optimizer_actor.step()
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(policy.critic1, policy.target_critic1)
        self.soft_update(policy.critic2, policy.target_critic2)
        self.soft_update(policy.critic3, policy.target_critic3)
        return actor_loss



    def train(self, policy: DSACPolicy, buffer: Union[ReplayBuffer, List[ReplayBuffer]]):
        train_info = {}  #TODO 需要添加一些观测数据


        if self.use_recurrent_policy:
            data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
        if self.use_transformer:
            data_generator = ReplayBuffer.recurrent_generator(buffer, self.num_mini_batch, self.data_chunk_length)
        else:
            raise NotImplementedError

        for sample in data_generator:
            self.dsac_update(policy, sample)


        for k in train_info.keys():
            train_info[k] /= self.num_mini_batch

        return train_info

