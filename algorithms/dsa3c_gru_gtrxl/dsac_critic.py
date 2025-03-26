import torch
import torch.nn as nn
from ..utils.mlp import MLPBase, MLPLayer
from ..utils.gru import GRULayer
from ..utils.gtrxl import GTrXL
from ..utils.utils import check


class DSACCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(DSACCritic, self).__init__()
        # 公共配置参数
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.use_transformer = args.use_transformer
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 公共维度定义
        self.obs_dim = obs_space.shape[0]
        self.action_dim = args.action_dim
        self.device = device

        # 公共模块初始化
        # (1) 特征提取模块
        self.base = MLPBase(
            obs_space,
            self.hidden_size,
            self.activation_id,
            self.use_feature_normalization
        )

        # (2) RNN模块
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(
                input_size,
                self.recurrent_hidden_size,
                self.recurrent_hidden_layers
            )
            input_size = self.rnn.output_size

        # (3) 值函数模块
        self._init_value_module(input_size)

        # GTrXL特有模块
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

        self.to(device)

    def _init_value_module(self, input_size):
        """初始化公共的值函数模块"""
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(
                input_size,
                self.act_hidden_size,
                self.activation_id
            )
        self.value_out = nn.Linear(input_size, 1)

    def forward(self, obs, rnn_states, masks, actions=None):
        # 输入预处理
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 公共处理流程
        critic_features = self.base(obs)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        # GRU处理 (原第一个版本特有)
        if self.use_recurrent_policy:
            if hasattr(self, 'gru'):
                critic_features = critic_features.unsqueeze(1)
                rnn_states = rnn_states.unsqueeze(0)
                critic_features, rnn_states = self.gru(critic_features, rnn_states)
                critic_features = critic_features.squeeze(1)
                rnn_states = rnn_states.squeeze(0)

        # GTrXL处理 (原第二个版本特有)
        if self.use_transformer:
            q_values = None
            if actions is not None and hasattr(self, 'gtrxl'):
                actions = check(actions).to(**self.tpdv)
                seq = torch.cat([obs, actions], dim=-1)
                transformer_out = self.gtrxl(seq)
                q_values = self.q_head(transformer_out[:, -1, :])

        # 公共值计算
        if hasattr(self, 'mlp'):
            critic_features = self.mlp(critic_features)
        values = self.value_out(critic_features)

        # 返回结果兼容两种版本
        return (values, q_values, rnn_states) if q_values is not None else (values, rnn_states)

    def get_q_values(self, obs, actions):
        """保留第二个版本特有方法"""
        if not hasattr(self, 'gtrxl'):
            raise NotImplementedError("GTrXL module not initialized")

        seq = torch.cat([obs, actions], dim=-1)
        transformer_out = self.gtrxl(seq)
        return self.q_head(transformer_out[:, -1, :])