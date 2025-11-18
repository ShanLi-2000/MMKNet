import math
import torch
import random
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class SyntheticNShot:

    def __init__(self, args, Is_linear=True, use_cuda=True):
        # 定义模型
        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.is_linear = Is_linear
        self.x_dim = 4
        self.y_dim = 2

        self.dt = 1
        self.const_F = torch.tensor([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]]).float().to(self.device)

        self.q2 = 10 ** (args.noise_q / 10)
        self.r2 = 10 ** (args.noise_r / 10)

        dt4 = 0.25 * self.dt ** 4
        dt3 = 0.5 * self.dt ** 3
        dt2 = self.dt ** 2
        # self.cov_q = self.q2 * torch.tensor([[dt4, 0, dt3, 0],
        #                                      [0, dt4, 0, dt3],
        #                                      [dt3, 0, dt2, 0],
        #                                      [0, dt3, 0, dt2]]).float()
        self.cov_q = self.q2 * torch.tensor([[dt4, 0, 0, 0],
                                             [0, dt4, 0, 0],
                                             [0, 0, dt2, 0],
                                             [0, 0, 0, dt2]]).float()

        if self.is_linear:
            self.cov_r = self.r2 * torch.tensor([[1, 0],
                                                 [0, 1]]).float()
        else:
            self.cov_r = self.r2 * torch.tensor([[1, 0],
                                                 [0, 0.0001]]).float()

        self.H = torch.eye(self.y_dim, self.x_dim).to(self.device)

        self.x_prev = torch.tensor([0., 0., 10., 0.], device=self.device).reshape(-1, 1)  # 生成数据的初始值
        self.init_state_filter = torch.tensor([0., 0., 20., 10.], device=self.device).reshape(-1, 1)  # 滤波器初始值设置
        self.ekf_cov = torch.zeros((self.x_dim, self.x_dim))

    def generate_data(self, seq_num, seq_len, mode="train", data_path=None, angular_velo_seq=None):
        if data_path is None:
            data_path = '../../Data/linear/'
        state = torch.zeros((seq_num, self.x_dim, seq_len), device=self.device)
        obs = torch.zeros((seq_num, self.y_dim, seq_len), device=self.device)
        ssm_angular_velo = torch.zeros((seq_num, seq_len), device=self.device)
        with torch.no_grad():
            x_prev = self.x_prev
            temp_l = torch.linalg.cholesky(self.cov_r)
            white_noise = torch.randn(seq_num, self.y_dim, seq_len)
            er = torch.bmm(temp_l.repeat(seq_num, 1, 1), white_noise).unsqueeze(-1).to(self.device)

            temp_l = torch.linalg.cholesky(self.cov_q)
            white_noise = torch.randn(seq_num, self.x_dim, seq_len)
            eq = torch.bmm(temp_l.repeat(seq_num, 1, 1), white_noise).unsqueeze(-1).to(self.device)

            x_prev = x_prev.repeat(seq_num, 1, 1)

            for b in range(seq_num):
                if angular_velo_seq is not None:
                    temp_angular_velo_seq = angular_velo_seq
                else:
                    temp_angular_velo_seq = self.random_angular_velo(seq_len)
                for i in range(seq_len):
                    temp_angular_velo = temp_angular_velo_seq[i]
                    xt = self.f(x_prev[b:b + 1], temp_angular_velo)
                    xt = torch.add(xt, eq[b:b + 1, :, i])
                    yt = self.g(xt)
                    yt = torch.add(yt, er[b:b + 1, :, i])
                    x_prev[b:b + 1] = xt.clone()
                    state[b, :, i] = torch.squeeze(xt, 2)
                    obs[b, :, i] = torch.squeeze(yt, 2)
                    ssm_angular_velo[b, i] = temp_angular_velo

        torch.save(state, data_path + mode + '/state.pt')
        torch.save(obs, data_path + mode + '/obs.pt')
        torch.save(ssm_angular_velo, data_path + mode + '/ssm_choose.pt')

    def f(self, x, w):
        F_add = self.get_F_parameters(x.shape[0], angular_velo=w)
        F = torch.add(F_add, self.const_F.repeat(x.shape[0], 1, 1))
        return torch.bmm(F, x)

    def g(self, x):
        if self.is_linear:
            return x[:, :2, :]
        else:
            x = x.squeeze(2)
            y1 = torch.sqrt((x[:, 0]) ** 2 + (x[:, 1]) ** 2).reshape(-1, 1)
            # 注意，y2该函数返回的是弧度而非角度，角度 = 弧度*180/pi
            y2 = torch.arctan2((x[:, 1]), (x[:, 0])).reshape(-1, 1)
            temp = torch.cat((y1, y2), dim=1)
            return temp.unsqueeze(dim=temp.dim())

    def Jacobian_f(self, x, w):
        F_add = self.get_F_parameters(x.shape[0], angular_velo=w)
        F = torch.add(F_add, self.const_F)
        return F

    def Jacobian_g(self, x):
        if self.is_linear:
            return torch.tensor([[1, 0, 0., 0.],
                                 [0, 1, 0., 0.]]).repeat(x.shape[0], 1, 1).to(x.device)
        else:
            temp_r = (x[:, 0]) ** 2 + (x[:, 1]) ** 2
            sqrt_r = torch.sqrt(temp_r)
            H11 = (x[:, 0]) / sqrt_r
            H12 = (x[:, 1]) / sqrt_r
            H21 = -(x[:, 1]) / temp_r
            H22 = (x[:, 0]) / temp_r
            temp = torch.zeros((x.shape[0], self.y_dim, self.x_dim))
            temp[:, 0, 0] = H11.squeeze()
            temp[:, 0, 1] = H12.squeeze()
            temp[:, 1, 0] = H21.squeeze()
            temp[:, 1, 1] = H22.squeeze()

            return temp.to(x.device)

    def get_F_parameters(self, seq_num, angular_velo):
        dt = float(self.dt)
        eps = 1e-8  # 稍小的阈值
        # move to device & ensure float
        if not torch.is_tensor(angular_velo):
            angular_velo = torch.tensor(float(angular_velo), device=self.device, dtype=torch.float32)
        else:
            angular_velo = angular_velo.to(self.device).float()

        # make angular_velo shape (seq_num,)
        if angular_velo.dim() == 0:
            angular_velo = angular_velo.unsqueeze(0).repeat(seq_num)
        elif angular_velo.numel() == 1 and seq_num > 1:
            angular_velo = angular_velo.view(1).repeat(seq_num)
        else:
            # ensure length matches seq_num or is broadcastable
            if angular_velo.shape[0] != seq_num:
                angular_velo = angular_velo.view(-1).repeat(seq_num)

        theta = angular_velo * dt  # shape (seq_num,)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        # use tensor eps on same device
        eps_t = torch.tensor(eps, device=self.device, dtype=torch.float32)

        # Taylor approximations for small omega:
        # sin(theta)/omega ≈ dt - omega^2 * dt^3 / 6
        sin_over_omega = torch.where(
            torch.abs(angular_velo) < eps_t,
            dt - (angular_velo ** 2) * (dt ** 3) / 6.0,
            sin_theta / angular_velo
        )

        # (cos(theta) - 1) / omega ≈ -0.5 * omega * dt^2
        cos_minus1_over_omega = torch.where(
            torch.abs(angular_velo) < eps_t,
            -0.5 * angular_velo * (dt ** 2),
            (cos_theta - 1.0) / angular_velo
        )

        # build F_temp (seq_num, 4, 4)
        F_temp = torch.zeros((seq_num, self.x_dim, self.x_dim), device=self.device, dtype=torch.float32)
        # fill blocks (broadcasting sin_over_omega etc along first dim)
        F_temp[:, 0, 2] = sin_over_omega.squeeze()
        F_temp[:, 0, 3] = cos_minus1_over_omega.squeeze()
        F_temp[:, 1, 2] = -cos_minus1_over_omega.squeeze()
        F_temp[:, 1, 3] = sin_over_omega.squeeze()
        F_temp[:, 2, 2] = cos_theta.squeeze()
        F_temp[:, 2, 3] = -sin_theta.squeeze()
        F_temp[:, 3, 2] = sin_theta.squeeze()
        F_temp[:, 3, 3] = cos_theta.squeeze()

        return F_temp

    def random_angular_velo(self, seq_len, p_cv=0.4, seg_min=50, seg_max=80, w_ranges=None):
        if w_ranges is None:
            # 定义几个典型的旋转角速度范围（单位：rad/s）
            w_ranges = [
                (math.pi / 30, math.pi / 20),  # 小转弯
                (math.pi / 20, math.pi / 10),  # 中等转弯
                (math.pi / 10, math.pi / 6)  # 大转弯
            ]

        w_seq = []
        t = 0
        while t < seq_len:
            # 随机片段长度
            seg_len = random.randint(seg_min, seg_max)
            seg_len = min(seg_len, seq_len - t)

            # 决定本段的角速度
            if random.random() < p_cv:
                w = 0.0  # CV 模式
            else:
                # 随机选一个角速度范围
                a, b = random.choice(w_ranges)
                base_w = random.uniform(a, b)  # 在范围内采样
                sign = random.choice([-1, 1])  # 左/右转弯
                w = sign * base_w

            # 填充该片段
            w_seq.extend([w] * seg_len)
            t += seg_len

        return torch.tensor(w_seq, dtype=torch.float32, device=self.device)
