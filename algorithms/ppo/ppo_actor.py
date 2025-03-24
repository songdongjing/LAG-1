import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 添加GRU相关参数
        if self.use_recurrent_policy:
            self.obs_dim = obs_space.shape[0]
            self.action_dim = act_space.shape[0]
            # 直接使用PyTorch原生GRU
            self.gru = nn.GRU(self.obs_dim, self.hidden_size, batch_first=True)
            self.fc = nn.Linear(self.hidden_size, self.action_dim * 2)  # 输出均值和标准差

        # (1) 特征提取模块
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)

        # (2) 原始RNN模块（如果需要保留GRULayer逻辑）
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            # 若需要兼容原有GRULayer，需调整输入维度
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size

        # (3) 动作输出模块
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 新增GRU逻辑
        if self.use_recurrent_policy:
            # 通过GRU处理原始观测
            gru_out, new_hidden = self.gru(obs.unsqueeze(1), rnn_states.unsqueeze(0))
            gru_out = gru_out.squeeze(1)
            mean, log_std = self.fc(gru_out).chunk(2, dim=-1)
            actor_features = mean
        else:
            # 原有特征提取流程
            actor_features = self.base(obs)

        # # 原有RNN逻辑（如果保留）
        # if self.use_recurrent_policy:
        #     actor_features, rnn_states = self.rnn(actor_features, new_hidden.squeeze(0), masks)

        # 动作生成
        actions, action_log_probs = self.act(actor_features, deterministic)
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # 评估时保持原有逻辑
        actor_features = self.base(obs)
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)
        return action_log_probs, dist_entropy