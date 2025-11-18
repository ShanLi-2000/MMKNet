import math
import random
import torch
import argparse
import numpy as np
from torch import optim
from filter import Filter
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from state_dict_learner import Learner
from gen_dataset import generate_random_segments
from Simulations.linear.linear_syntheticNShot import SyntheticNShot


def set_seed(seed=1111):
    random.seed(seed)  # Python内建随机数
    np.random.seed(seed)  # Numpy随机数
    torch.manual_seed(seed)  # CPU随机数
    torch.cuda.manual_seed(seed)  # GPU随机数
    torch.cuda.manual_seed_all(seed)  # 多GPU随机数
    torch.backends.cudnn.deterministic = True  # cuDNN算法确定性
    torch.backends.cudnn.benchmark = False  # 禁止算法搜索带来的随机性


# 5555/777/3(ct_begin) 4396(cv_begin)
set_seed(5555)


def main(args):
    print(args)

    is_generate_data = False
    is_train = False
    args.use_cuda = True
    if args.use_cuda:
        if torch.cuda.is_available():
            print("Using GPU")
            device = torch.device('cuda')
        else:
            raise Exception("No GPU found, please set args.use_cuda = False")
    else:
        print("Using CPU")
        device = torch.device('cpu')

    linear_model = SyntheticNShot(args, Is_linear=True)

    if is_generate_data:
        linear_model.generate_data(20, 10000, data_path='./Data/linear/')
    linear_model.generate_data(1, 500, mode='test', data_path='./Data/linear/')
    state_train = torch.load('./Data/linear/train/state.pt', weights_only=True)
    obs_train = torch.load('./Data/linear/train/obs.pt', weights_only=True)
    ssm_angular_velo_train = torch.load('./Data/linear/train/ssm_choose.pt', weights_only=True)

    state_test = torch.load('./Data/linear/test/state.pt', weights_only=True)
    obs_test = torch.load('./Data/linear/test/obs.pt', weights_only=True)
    ssm_angular_velo_test = torch.load('./Data/linear/test/ssm_choose.pt', weights_only=True)

    batches = generate_random_segments(
        state_train, obs_train, ssm_angular_velo_train, num_batches=args.num_batches, batch_size=args.batch_size,
        L_min=args.L_min, L_max=args.L_max, N_check=args.N_check)

    my_filter = Filter(args, linear_model)

    MMKNet = Learner(y_dim=linear_model.y_dim, d_model=64, nhead=4, num_layers=2).to(device)
    MMKNet_mismatch = Learner(y_dim=linear_model.y_dim, d_model=64, nhead=4, num_layers=2).to(device)

    cal_num_param = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("MMKNet所有需要梯度更新的参数的总数量" + str(cal_num_param(MMKNet)))

    # training neural networks
    if is_train:

        # Training MMKNet
        my_optimizer = optim.Adam(MMKNet.parameters(), lr=args.update_lr, weight_decay=args.weight_decay)
        for j in range(args.epoch):
            random.shuffle(batches)
            for i, batch in enumerate(batches):
                loss, filtering_loss = my_filter.compute_x_post(batch, MMKNet)
                my_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(MMKNet.parameters(), 5)
                my_optimizer.step()
                if i % 25 == 0:
                    with torch.no_grad():
                        loss_ekf = my_filter.EKF(batch)
                        loss_ekf_target = my_filter.EKF(batch, is_omega=True)
                    print(str(j) + 'th epoch ' + str(i) + 'th MMKNet training:')
                    print('loss: ' + str(loss.item()))
                    print('MSE (dB): ' + str((10 * torch.log10(filtering_loss)).item()))
                    print('EKF with constant CV SSM MSE (dB): ' + str(loss_ekf.item()))
                    print('EKF with accurate omega MSE (dB): ' + str(loss_ekf_target.item()))
        torch.save(MMKNet.state_dict(), './Model/linear/MMKNet.pt')
    else:
        temp_net = torch.load('./Model/linear/MMKNet.pt')
        MMKNet.load_state_dict(temp_net)

        temp_net_mismatch = torch.load('./Model/linear/MMKNet_mismatch.pt')
        MMKNet_mismatch.load_state_dict(temp_net_mismatch)

    # testing the filtering result
    with torch.no_grad():
        # MMKNet
        state_filtering, angular_velo_estimate = my_filter.compute_x_post_qry(state_test, obs_test, ssm_angular_velo_test, MMKNet)
        loss_state_filtering = my_filter.loss_fn(state_filtering[:, :2, :], state_test[:, :2, :])
        loss_angular_estimate = my_filter.loss_fn(angular_velo_estimate, ssm_angular_velo_test)

        # MMKNet_mismatch
        state_filtering_mismatch, angular_velo_estimate_mismatch = my_filter.compute_x_post_qry(state_test, obs_test, ssm_angular_velo_test, MMKNet_mismatch)
        loss_state_filtering_mismatch = my_filter.loss_fn(state_filtering_mismatch[:, :2, :], state_test[:, :2, :])
        loss_angular_estimate_mismatch = my_filter.loss_fn(angular_velo_estimate_mismatch, ssm_angular_velo_test)

        temp_batch = {
            "state": state_test,
            "obs": obs_test,
            "omega": ssm_angular_velo_test
        }

        # KF with constant CV model
        state_filtering_KF = my_filter.EKF(temp_batch, is_trajectory=True)
        loss_state_filtering_KF = my_filter.loss_fn(state_filtering_KF[:, :2, :], state_test[:, :2, :])

        # KF with True Omega model
        state_filtering_KF_True = my_filter.EKF(temp_batch, is_omega=True, is_trajectory=True)
        loss_state_filtering_KF_True = my_filter.loss_fn(state_filtering_KF_True[:, :2, :], state_test[:, :2, :])

    print("EKF with True Omega filtering loss: " + str(loss_state_filtering_KF_True.item()))
    print("MMKNet angular loss: " + str(loss_angular_estimate.item()))
    print("MMKNet filtering loss: " + str(loss_state_filtering.item()))
    print("MMKNet (mismatch) angular loss: " + str(loss_angular_estimate_mismatch.item()))
    print("MMKNet (mismatch) filtering loss: " + str(loss_state_filtering_mismatch.item()))
    print("EKF(CV) filtering loss: " + str(loss_state_filtering_KF.item()))
    state_test = state_test.cpu()
    state_filtering = state_filtering.cpu()
    state_filtering_mismatch = state_filtering_mismatch.cpu()

    start_point = 250
    end_point = 300

    plt.plot(state_test[0, 0, start_point:end_point], state_test[0, 1, start_point:end_point], color='black',
             label="state", linestyle='solid', markerfacecolor='none',
             markeredgecolor='black', markersize=10)
    plt.plot(state_filtering_mismatch[0, 0, start_point:end_point], state_filtering_mismatch[0, 1, start_point:end_point],
             color='#9467BD',
             label="MMKNet (mismatch)", linestyle='dashed', markerfacecolor='none',
             markeredgecolor='black', markersize=10)
    plt.plot(state_filtering[0, 0, start_point:end_point], state_filtering[0, 1, start_point:end_point],
             color='#00ff00',
             label="MMKNet", linestyle='dashed', markerfacecolor='none',
             markeredgecolor='black', markersize=10)

    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
    fontsize = 12
    plt.xlabel('X', fontsize=fontsize, fontweight='bold')
    plt.ylabel('Y', fontsize=fontsize, fontweight='bold')
    plt.legend(fontsize=fontsize, loc=0, prop={'size': 11})
    plt.grid()
    plt.tight_layout()

    # # 插入局部放大图
    # ax_inset = inset_axes(plt.gca(), width="40%", height="35%")  # 插入小图
    # ax_inset = plt.axes([0.62, 0.2, 0.35, 0.3])  # [x, y, width, height]
    # start_idx = 230  # 选择放大最后几个点
    # end_idx = 250
    # markersize = 5
    #
    # # 放大数据绘制到局部图
    # ax_inset.plot(state_test[0, 0, start_idx:end_idx], state_test[0, 1, start_idx:end_idx], color='black',
    #               label="state", linestyle='solid', markerfacecolor='none',
    #               markeredgecolor='black', markersize=10)
    # ax_inset.plot(state_filtering_mismatch[0, 0, start_idx:end_idx], state_filtering_mismatch[0, 1, start_idx:end_idx],
    #               color='#9467BD',
    #               label="MMKNet (mismatch)", linestyle='dashed', markerfacecolor='none',
    #               markeredgecolor='black', markersize=10)
    # ax_inset.plot(state_filtering[0, 0, start_idx:end_idx], state_filtering[0, 1, start_idx:end_idx],
    #               color='#00ff00',
    #               label="MMKNet", linestyle='dashed', markerfacecolor='none',
    #               markeredgecolor='black', markersize=10)
    # # 局部图参数设置
    # ax_inset.grid()
    # ax_inset.tick_params(labelsize=8)
    #
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(ssm_angular_velo_test[0].cpu().numpy(), label='True Angular Velocity', color='black', linewidth=2)
    plt.plot(angular_velo_estimate_mismatch[0].cpu().numpy(), label='Estimated Angular Velocity (mismatch)', color='#9467BD',
             linestyle='dashed')
    plt.plot(angular_velo_estimate[0].cpu().numpy(), label='Estimated Angular Velocity', color='green',
             linestyle='dashed')
    plt.xlabel('Time Step', fontsize=12, fontweight='bold')
    plt.ylabel('Angular Velocity [rad/s]', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
    argparser.add_argument('--batch_size', type=int, help='batch size for training', default=16)
    argparser.add_argument('--update_lr', type=float, help='update learning rate', default=0.001)
    argparser.add_argument('--weight_decay', type=float, help='update weight decay', default=0)
    argparser.add_argument('--noise_q', type=float, help='process noise in dB level', default=5)
    argparser.add_argument('--noise_r', type=float, help='measurement noise in dB level', default=15)
    argparser.add_argument('--L_min', type=int, help='minimum length for training sequence', default=40)
    argparser.add_argument('--L_max', type=int, help='maximum length for training sequence', default=120)
    argparser.add_argument('--num_batches', type=int, help='number of batches', default=500)
    argparser.add_argument('--N_check', type=int, help='sliding window for transformer', default=8)
    argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

    args = argparser.parse_args()

    main(args)
