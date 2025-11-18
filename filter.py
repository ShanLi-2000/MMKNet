import torch
from torch import nn


class Filter:
    def __init__(self, args, model):
        self.update_lr = args.update_lr

        self.model = model
        self.args = args
        if args.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise Exception("No GPU found, please set args.use_cuda = False")
        else:
            self.device = torch.device('cpu')

        self.loss_fn = nn.MSELoss()

        self.cov_q = model.cov_q
        self.cov_r = model.cov_r

        self.state_post = self.model.init_state_filter
        self.ekf_state_post = self.model.init_state_filter

        self.obs_past = torch.zeros((model.y_dim, 1), device=self.device)
        self.obs_past_past = torch.zeros((model.y_dim, 1), device=self.device)

        self.cov_post = self.model.ekf_cov.to(self.device)

        self.ekf_cov = self.model.ekf_cov.to(self.device)
        self.ekf_cov_post = self.ekf_cov.detach().clone()

        self.batch_size = args.batch_size

        self.beta = 0
        self.training_first = True
        self.omega = torch.tensor(0., device=self.device)

        self.state_predict_train = self.model.init_state_filter
        self.state_pred_past = self.model.init_state_filter
        self.state_post_past = self.model.init_state_filter

    def compute_x_post(self, current_batch, task_net=None):
        if task_net is None:
            raise ValueError("Please define a neural network")
        else:
            temp_net = task_net
        self.reset_net()

        batch_state = current_batch['state']
        batch_obs = current_batch['obs']
        batch_angular_velo = current_batch['omega']

        seq_num, y_dim, seq_len = batch_obs.shape

        self.cov_q = self.model.cov_q.repeat(seq_num, 1, 1).to(self.device)
        self.cov_r = self.model.cov_r.repeat(seq_num, 1, 1).to(self.device)

        state_filtering = torch.zeros_like(batch_state)
        angular_velo_estimate = torch.zeros_like(batch_angular_velo)

        # 软启动
        # self.state_post = self.model.init_state_filter.reshape((1, self.model.x_dim, 1)).\
        #     repeat(seq_num, 1, 1).to(batch_state.device)
        self.state_post = batch_state[:, :, 1].unsqueeze(-1)
        if self.cov_post.ndim != self.state_post.ndim and self.cov_post.ndim == 2:
            self.cov_post = self.cov_post.repeat(seq_num, 1, 1)

        residual_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        velo_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        acc_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        count = self.args.N_check + 2 - 1
        for i in range(2, count):
            state_filtering[:, :, i], residual_temp = self.warm_up_filtering(batch_obs[:, :, i].unsqueeze(-1))
            velo_temp = batch_obs[:, :, i].unsqueeze(0) - batch_obs[:, :, i - 1].unsqueeze(0)
            acc_temp = velo_seq[i - 2] - (batch_obs[:, :, i].unsqueeze(0) - batch_obs[:, :, i - 1].unsqueeze(0))
            residual_seq = torch.cat((residual_seq, residual_temp.unsqueeze(0)), dim=0)
            velo_seq = torch.cat((velo_seq, velo_temp), dim=0)
            acc_seq = torch.cat((acc_seq, acc_temp), dim=0)

        self.obs_past = batch_obs[:, :, self.args.N_check].unsqueeze(dim=2)
        for i in range(count, seq_len):
            state_filtering[:, :, i], angular_velo_estimate[:, i], residual_seq, velo_seq, acc_seq = \
                self.filtering(batch_obs[:, :, i].unsqueeze(dim=2), residual_seq, velo_seq, acc_seq, temp_net)

        self.reset_net()

        loss1 = self.loss_fn(state_filtering[:, :2, count:], batch_state[:, :2, count:])
        loss2 = self.loss_fn(angular_velo_estimate[:, count:], batch_angular_velo[:, count:])
        loss = self.beta * loss1 + (1 - self.beta) * loss2

        return loss, loss1

    def compute_x_post_qry(self, state, obs, angular_velo, task_net=None):  # obs:[seq_num, y_dim, seq_len]
        if task_net is None:
            raise ValueError("Please define a neural network")
        else:
            temp_net = task_net
        self.reset_net()

        seq_num, y_dim, seq_len = obs.shape

        self.cov_q = self.model.cov_q.repeat(seq_num, 1, 1).to(self.device)
        self.cov_r = self.model.cov_r.repeat(seq_num, 1, 1).to(self.device)

        state_filtering = torch.zeros_like(state)
        angular_velo_estimate = torch.zeros_like(angular_velo)

        # 软启动
        # self.state_post = self.model.init_state_filter.reshape((1, self.model.x_dim, 1)).\
        #     repeat(seq_num, 1, 1).to(batch_state.device)
        self.state_post = state[:, :, 1].unsqueeze(-1)
        if self.cov_post.ndim != self.state_post.ndim and self.cov_post.ndim == 2:
            self.cov_post = self.cov_post.repeat(seq_num, 1, 1)

        residual_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        velo_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        acc_seq = torch.zeros((1, seq_num, y_dim), device=self.device)
        count = self.args.N_check + 2 - 1
        for i in range(2, count):
            state_filtering[:, :, i], residual_temp = self.warm_up_filtering(obs[:, :, i].unsqueeze(-1))
            velo_temp = obs[:, :, i].unsqueeze(0) - obs[:, :, i - 1].unsqueeze(0)
            acc_temp = velo_seq[i - 2] - (obs[:, :, i].unsqueeze(0) - obs[:, :, i - 1].unsqueeze(0))
            residual_seq = torch.cat((residual_seq, residual_temp.unsqueeze(0)), dim=0)
            velo_seq = torch.cat((velo_seq, velo_temp), dim=0)
            acc_seq = torch.cat((acc_seq, acc_temp), dim=0)

        self.obs_past = obs[:, :, self.args.N_check].unsqueeze(dim=2)
        for i in range(count, seq_len):
            state_filtering[:, :, i], angular_velo_estimate[:, i], residual_seq, velo_seq, acc_seq = \
                self.filtering(obs[:, :, i].unsqueeze(dim=2), residual_seq, velo_seq, acc_seq, temp_net)

        self.reset_net()

        loss1 = self.loss_fn(state_filtering[:, :2, count:], state[:, :2, count:])
        loss2 = self.loss_fn(angular_velo_estimate[:, count:], angular_velo[:, count:])

        return state_filtering, angular_velo_estimate

    def warm_up_filtering(self, observation):
        x_last = self.state_post
        x_predict = self.model.f(x_last, w=0)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        F_jacob = self.model.Jacobian_f(x_last, w=0)
        H_jacob = self.model.Jacobian_g(x_predict)

        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.cov_post), torch.transpose(F_jacob, 1, 2))) + self.cov_q
        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2)) + self.cov_r)
        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)

        x_post = x_predict + torch.bmm(K_gain, residual)

        cov_post = torch.bmm((torch.eye(self.model.x_dim, device=self.device) - torch.bmm(K_gain, H_jacob)), cov_pred)

        self.state_post = x_post
        self.cov_post = cov_post

        return x_post.squeeze(-1), residual.squeeze(-1)

    def filtering(self, observation, residual_seq, velo_seq, acc_seq, task_net):
        x_last = self.state_post
        # 根据前一时刻的角速度进行预测，得到当前时刻的残差和差分，进而得到当前时刻的网络输入。
        if self.training_first:
            self.omega = torch.zeros((observation.shape[0], 1), device=self.device)
        x_predict = self.model.f(x_last, w=self.omega)
        y_predict = self.model.g(x_predict)
        residual = observation - y_predict
        residual_seq = torch.cat((residual_seq[1:], residual.permute(2, 0, 1)), dim=0)
        velo_seq = torch.cat((velo_seq[1:], (observation - self.obs_past).permute(2, 0, 1)), dim=0)
        acc_seq = torch.cat((acc_seq[1:], (velo_seq[-1] - velo_seq[-2]).unsqueeze(0)), dim=0)

        # 网络输出当前时刻的角速度Omega
        self.omega = task_net(residual_seq, velo_seq, acc_seq)

        # 再次执行预测步，并执行更新步
        x_predict = self.model.f(x_last, w=self.omega)
        y_predict = self.model.g(x_predict)
        residual = observation - y_predict
        residual_seq[-1] = residual.squeeze()  # 更新当前时刻残差 (是否需要更新残差序列中最后一个被纠正的残差)

        F_jacob = self.model.Jacobian_f(x_last, w=self.omega)
        H_jacob = self.model.Jacobian_g(x_predict)

        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.cov_post), torch.transpose(F_jacob, 1, 2))) + self.cov_q
        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2)) + self.cov_r)
        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)

        x_post = x_predict + torch.bmm(K_gain, residual)

        cov_post = torch.bmm((torch.eye(self.model.x_dim, device=self.device) - torch.bmm(K_gain, H_jacob)), cov_pred)

        self.training_first = False
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.cov_post = cov_post.detach().clone()

        return x_post.clone().squeeze(-1), self.omega.clone().squeeze(-1), residual_seq, velo_seq, acc_seq

    def reset_net(self):
        self.training_first = True
        self.cov_post = self.model.ekf_cov.to(self.device)
        self.state_post = self.model.init_state_filter

    def EKF(self, current_batch, is_omega=False, is_trajectory=False):
        batch_state = current_batch['state']
        batch_obs = current_batch['obs']
        batch_angular_velo = current_batch['omega']

        seq_num, y_dim, seq_len = batch_obs.shape

        self.reset_ekf()
        self.cov_q = self.model.cov_q.repeat(seq_num, 1, 1).to(self.device)
        self.cov_r = self.model.cov_r.repeat(seq_num, 1, 1).to(self.device)

        state_filtering = torch.zeros_like(batch_state)

        if is_omega:
            temp_omega = batch_angular_velo
        else:
            temp_omega = torch.zeros_like(batch_angular_velo)
        self.ekf_state_post = batch_state[:, :, 0].unsqueeze(-1)
        if self.ekf_cov_post.ndim != self.ekf_state_post.ndim and self.ekf_cov_post.ndim == 2:
            self.ekf_cov_post = self.ekf_cov_post.repeat(seq_num, 1, 1)
        for i in range(1, seq_len):
            state_filtering[:, :, i] = self.ekf_filtering(batch_obs[:, :, i].unsqueeze(dim=2),
                                                          temp_omega[:, i].unsqueeze(dim=1))
        self.reset_ekf()

        if is_trajectory:
            return state_filtering
        else:
            loss = self.loss_fn(state_filtering[:, :2, 1:], batch_state[:, :2, 1:])
            # self.state_EKF = state_filtering.clone().detach()

            return 10 * torch.log10(loss)

    def ekf_filtering(self, observation, omega):
        x_last = self.ekf_state_post
        x_predict = self.model.f(x_last, w=omega)

        y_predict = self.model.g(x_predict)
        residual = observation - y_predict

        F_jacob = self.model.Jacobian_f(x_last, w=omega)
        H_jacob = self.model.Jacobian_g(x_predict)

        cov_pred = (torch.bmm(torch.bmm(F_jacob, self.ekf_cov_post), torch.transpose(F_jacob, 1, 2))) + self.cov_q
        temp = torch.linalg.inv(torch.bmm(torch.bmm(H_jacob, cov_pred), torch.transpose(H_jacob, 1, 2)) + self.cov_r)
        K_gain = torch.bmm(torch.bmm(cov_pred, torch.transpose(H_jacob, 1, 2)), temp)  # torch.linalg.inv 计算矩阵的倒数

        x_post = x_predict + torch.bmm(K_gain, residual)
        cov_post = torch.bmm((torch.eye(self.model.x_dim, device=observation.device) - torch.bmm(K_gain, H_jacob)),
                             cov_pred)

        self.ekf_state_post = x_post.detach().clone()
        self.ekf_cov_post = cov_post.detach().clone()

        return x_post.clone().squeeze()

    def reset_ekf(self):
        self.ekf_state_post = self.model.init_state_filter.to(self.device)
        self.ekf_cov_post = self.model.ekf_cov.to(self.device)
