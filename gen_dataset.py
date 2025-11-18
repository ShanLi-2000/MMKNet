import torch
import random
import math
import numpy as np


def generate_random_segments(state, obs, ssm_angular_velo, num_batches=500, batch_size=16,
                             L_min=40, L_max=120, N_check=10):
    num_traj, x_dim, seq_total = state.shape
    _, y_dim, _ = obs.shape

    batches = []

    for _ in range(num_batches):
        # 每个 batch 的长度固定，但在 L_min~L_max 随机
        L = random.randint(L_min, L_max)
        state_batch = []
        obs_batch = []
        omega_batch = []

        while len(state_batch) < batch_size:
            # 随机选一条轨迹
            traj_idx = random.randint(0, num_traj - 1)

            # 随机起始点（保证片段完整）
            start = random.randint(0, seq_total - L)

            w_seg = ssm_angular_velo[traj_idx, start:start + L]
            # 检查前 N_check 帧角速度是否一致
            if torch.allclose(w_seg[:N_check], w_seg[0].expand(N_check), atol=1e-6):
                state_batch.append(state[traj_idx, :, start:start + L])
                obs_batch.append(obs[traj_idx, :, start:start + L])
                omega_batch.append(w_seg)
            # 否则丢弃，重新采样

        # 拼接成 batch
        state_batch = torch.stack(state_batch, dim=0)  # [B, x_dim, L]
        obs_batch = torch.stack(obs_batch, dim=0)      # [B, y_dim, L]
        omega_batch = torch.stack(omega_batch, dim=0)  # [B, L]

        batches.append({
            "state": state_batch,
            "obs": obs_batch,
            "omega": omega_batch
        })

    return batches
