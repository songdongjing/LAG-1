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

        self.target_critic1 = DSACCritic(args, self.obs_space, self.device)
        self.target_critic2 = DSACCritic(args, self.obs_space, self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.optimizer_actor = torch.optim.Adam([
            {'params': self.actor.parameters()},
        ], lr=self.lr)

        self.optimizer_critic1 = torch.optim.Adam([
            {'params': self.critic1.parameters()},
        ], lr=self.lr)

        self.optimizer_critic2 = torch.optim.Adam([
            {'params': self.critic2.parameters()},
        ], lr=self.lr)
    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, masks):
        """
        Returns:
            values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks)
        values1, rnn_states_critic = self.critic1(obs, rnn_states_critic, masks)
        values2, rnn_states_critic = self.critic2(obs, rnn_states_critic, masks)
        values = torch.minimum(values1, values2)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks):
        """
        Returns:
            values
        """
        values1, _ = self.critic1(obs, rnn_states_critic, masks)
        values2, _ = self.critic2(obs, rnn_states_critic, masks)
        values = torch.minimum(values1, values2)
        return values


    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, action, masks, active_masks=None):
        """
        Returns:
            values, action_log_probs, dist_entropy
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, active_masks)
        values1, _ = self.critic1(obs, rnn_states_critic, masks)
        values2, _ = self.critic2(obs, rnn_states_critic, masks)
        values = torch.minimum(values1, values2)
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
        self.target_critic1.train()
        self.target_critic2.train()
    def prep_rollout(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
    def copy(self):
        return DSACPolicy(self.args, self.obs_space, self.act_space, self.device)
