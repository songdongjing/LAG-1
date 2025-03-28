import torch
from .dsac_actor import DSACActor
from .dsac_critic import DSACCritic
import numpy as np

class DSACPolicy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):

        self.args = args
        self.device = device
        # optimizer config
        self.lr = args.lr

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = DSACActor(args, self.obs_space, self.act_space, self.device)

        self.critic1 = DSACCritic(args, self.obs_space, self.device)
        self.critic2 = DSACCritic(args, self.obs_space, self.device)
        self.critic3= DSACCritic(args, self.obs_space, self.device)

        self.target_critic1 = DSACCritic(args, self.obs_space, self.device)
        self.target_critic2 = DSACCritic(args, self.obs_space, self.device)
        self.target_critic3 = DSACCritic(args, self.obs_space, self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic3.load_state_dict(self.critic3.state_dict())

        self.optimizer_actor = torch.optim.Adam([
            {'params': self.actor.parameters()},
        ], lr=self.lr)

        self.optimizer_critic1 = torch.optim.Adam([
            {'params': self.critic1.parameters()},
        ], lr=self.lr)

        self.optimizer_critic2 = torch.optim.Adam([
            {'params': self.critic2.parameters()},
        ], lr=self.lr)

        self.optimizer_critic3 = torch.optim.Adam([
            {'params': self.critic3.parameters()},
        ], lr=self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values1, rnn_states_critic = self.critic1(obs, rnn_states_critic, masks)
        values2, rnn_states_critic = self.critic2(obs, rnn_states_critic, masks)
        values3, rnn_states_critic = self.critic3(obs, rnn_states_critic, masks)
        values = self.average_of_two_smallest(values1, values2, values3)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values1, _ = self.critic1(obs, rnn_states_critic, masks)
        values2, _ = self.critic2(obs, rnn_states_critic, masks)
        values3, _ = self.critic3(obs, rnn_states_critic, masks)
        values = self.average_of_two_smallest(values1, values2, values3)
        return values

    def average_of_two_smallest(self, val1, val2, val3):
        """
        计算三个张量中最小两个值的平均值。

        参数:
            val1, val2, val3 (torch.Tensor): 形状为 (batch_size, 1) 或类似的三张量。

        返回:
            torch.Tensor: 形状为 (batch_size, 1) 的张量，包含最小两个值的平均值。
        """
        stacked = torch.stack([val1, val2, val3], dim=0)
        two_smallest, _ = torch.topk(stacked, k=2, dim=0, largest=False)
        return torch.mean(two_smallest, dim=0)


    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values1, _ = self.critic1(obs, rnn_states_critic, masks)
        values2, _ = self.critic2(obs, rnn_states_critic, masks)
        values3, _ = self.critic3(obs, rnn_states_critic, masks)
        values = self.average_of_two_smallest(values1, values2, values3)
        return values, action_log_probs, dist_entropy




    def act(self, obs, rnn_states_actor, masks, deterministic=False):
        """
        Returns:
            actions, rnn_states_actor
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, deterministic)
        return actions, rnn_states_actor

    def prep_training(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.critic3.train()
        self.target_critic1.train()
        self.target_critic2.train()
        self.target_critic3.train()
    def prep_rollout(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.critic3.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.target_critic3.eval()
    def copy(self):
        return DSACPolicy(self.args, self.obs_space, self.act_space, self.device)
