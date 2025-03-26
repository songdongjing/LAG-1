import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class DSACActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(DSACActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.use_transformer = args.use_transformer
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_prior = args.use_prior
        ##添加GRU
        if self.use_recurrent_policy:
            # Define observation and action dimensions
            self.obs_dim = obs_space.shape[0]  # Assuming obs_space is a Box or similar
            self.action_dim = act_space.shape[0]  # Assuming act_space is a Box or similar

            # Add GRU layer
            self.gru = nn.GRU(self.obs_dim, self.hidden_size, batch_first=True)

            # Add fully connected layer
            self.fc = nn.Linear(self.hidden_size, self.action_dim * 2)  # 输出均值和标准差

        # (1) feature extraction module
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
        # (2) rnn module
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)



        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11])  # unit degree
            distance = obs[:, 13] * 10000  # unit m
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance <= 12000] = 6
            alpha0[distance <= 8000] = 10
            beta0[attack_angle <= 45] = 6
            beta0[attack_angle <= 22.5] = 3
        ##添加GRU
        if self.use_recurrent_policy:
            # Add GRU logic
            gru_out, new_hidden = self.gru(obs, rnn_states)  # Pass rnn_states as hidden_state
            mean, log_std = self.fc(gru_out).chunk(2, dim=-1)  # Split the output into mean and log_std
            actor_features = mean  # Use the mean from the GRU as actor_features
        else:
            actor_features = self.base(obs)

        # if self.use_recurrent_policy:
        #    actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            actions, action_log_probs = self.act(actor_features, deterministic, alpha0=alpha0, beta0=beta0)
        else:
            actions, action_log_probs = self.act(actor_features, deterministic)

        if self.use_recurrent_policy:
            return actions, action_log_probs, new_hidden  # Return new_hidden as rnn_states
        else:
            return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if self.use_prior:
            # prior knowledage for controling shoot missile
            attack_angle = torch.rad2deg(obs[:, 11])  # unit degree
            distance = obs[:, 13] * 10000  # unit m
            alpha0 = torch.full(size=(obs.shape[0], 1), fill_value=3).to(**self.tpdv)
            beta0 = torch.full(size=(obs.shape[0], 1), fill_value=10).to(**self.tpdv)
            alpha0[distance <= 12000] = 6
            alpha0[distance <= 8000] = 10
            beta0[attack_angle <= 45] = 6
            beta0[attack_angle <= 22.5] = 3

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.use_prior:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks,
                                                                       alpha0=alpha0, beta0=beta0)
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy
