import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.gtrxl import GTrXL
from ..utils.utils import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # 新增参数
        self.use_transformer = args.use_transformer
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 原网络配置
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.obs_dim = obs_space.shape[0]
        self.action_dim = args.action_dim if hasattr(args, 'action_dim') else 0

        # (1) 特征提取模块
        self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)

        # (2) RNN模块
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size

        # (3) Transformer模块
        if self.use_transformer:
            if hasattr(args, 'seq_len'):
                self.seq_len = args.seq_len
                self.gtrxl = GTrXL(
                    d_model=self.obs_dim + self.action_dim,
                    n_layers=4,
                    n_heads=8,
                    seq_len=self.seq_len
                )
                self.q_head = nn.Linear(self.obs_dim + self.action_dim, 1)

        # (4) 值函数模块
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks, actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        # RNN处理
        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        # Transformer处理
        if self.use_transformer:
            q_values = None
            if self.use_transformer and actions is not None:
                actions = check(actions).to(**self.tpdv)
                seq = torch.cat([obs, actions], dim=-1)
                transformer_out = self.gtrxl(seq)
                q_values = self.q_head(transformer_out[:, -1, :])

        # 值函数计算
        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)
        values = self.value_out(critic_features)

        return (values, q_values, rnn_states) if q_values is not None else (values, rnn_states)

    def get_q_values(self, obs, actions):
        """GTrXL专用的Q值计算方法"""
        if not self.use_transformer:
            raise NotImplementedError("GTrXL module not initialized")

        seq = torch.cat([obs, actions], dim=-1)
        transformer_out = self.gtrxl(seq)
        return self.q_head(transformer_out[:, -1, :])