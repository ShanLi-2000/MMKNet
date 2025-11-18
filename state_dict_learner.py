import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class Learner(nn.Module):

    def __init__(self, y_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, w_max=1.0, max_len=500):
        super().__init__()
        self.y_dim = y_dim
        self.input_dim = y_dim * 3
        self.d_model = d_model
        self.w_max = w_max

        # 输入映射层
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, residual_seq, velo_seq, acc_seq):
        # 拼接输入特征
        x = torch.cat([residual_seq, velo_seq, acc_seq], dim=2)  # (N_check, seq_num, 3*y_dim)
        x = x.squeeze(-1)  # (N_check, seq_num, 3*y_dim)
        x = x.permute(1, 0, 2)  # (seq_num, N_check, 3*y_dim)

        # 映射到 d_model
        x = self.input_proj(x)  # (seq_num, N_check, d_model)

        # 加入位置编码
        x = self.pos_encoder(x)

        # Transformer 编码
        x = self.encoder(x)  # (seq_num, N_check, d_model)

        # 使用最后一个时间步的隐藏状态作为输出
        x_last = x[:, -1, :]  # (seq_num, d_model)

        # 输出角速度
        omega_hat = self.fc_out(x_last)  # (seq_num, 1)
        omega_hat = self.w_max * torch.tanh(omega_hat)

        return omega_hat
