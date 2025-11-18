import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from filter import Filter
from linear_syntheticNShot import SyntheticNShot
from state_dict_learner import Learner
from torch.distributions.multivariate_normal import MultivariateNormal
from DANSE.DANSE_filter import DANSE_Filter

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number for danse', default=100)
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_spt_test', type=int, help='k shot for support set test', default=8)
argparser.add_argument('--q_qry', type=int, help='q shot for query set', default=15)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=16)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.0011)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)
argparser.add_argument('--batch_size', type=int, help='batch size for train MAML', default=25)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=16)
argparser.add_argument('--task_num_train', type=int, help='task num for train', default=25)
argparser.add_argument('--task_num_test', type=int, help='task num for test', default=15)
argparser.add_argument('--use_cuda', type=int, help='use GPU to accelerate training', default=False)

args = argparser.parse_args()
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

Generate_data = False
Train = False
data_path = '../../MAML_data/linear/'
model = SyntheticNShot(batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_shot_test=args.k_spt_test,
                       q_query=args.q_qry,
                       data_path=data_path,
                       use_cuda=args.use_cuda,
                       Is_GenData=False)
my_filter = Filter(args, model, is_linear_net=True)
loss_fn = torch.nn.MSELoss()
random_net = Learner(model.x_dim, model.y_dim, args, is_linear_net=True).to(device)
# random_net = torch.load('../../MAML_model/linear/random_init_net.pt')
pretrained_net = torch.load('../../MAML_model/linear/model_trajectory/pre_training_KalmanNet.pt')
my_dict = torch.load('../../MAML_model/linear/basenet_11000.pt')  # 如果是加载整个网络，则不需要显示创建网络

danse_filter = DANSE_Filter(args, model)
danse_dict = danse_filter.danse_net.state_dict()

torch.manual_seed(3333)
np.random.seed(3333)
# 生成测试数据
seq_num = 300
seq_len = 50
noise_q = [0.15]
v = [-10., -5., 0., 5., 10., 15., 20.]
noise_num = len(v) * len(noise_q)
if Generate_data:

    state = torch.zeros((noise_num, seq_num, model.x_dim, seq_len), device=device)
    obs = torch.zeros((noise_num, seq_num, model.y_dim, seq_len), device=device)
    j = 0
    for q2 in noise_q:

        for elem in v:

            cov_q = q2 * torch.eye(model.x_dim)
            r2 = q2 * 10 ** (-elem / 10)
            cov_r = r2 * torch.eye(model.y_dim)
            print("Generating Data: q2: " + str(q2) + "  r2: " + str(r2))
            state_mtx = torch.zeros((seq_num, model.x_dim, seq_len), device=device)  # 初始化状态矩阵
            obs_mtx = torch.zeros((seq_num, model.y_dim, seq_len), device=device)  # 初始化观测矩阵
            # x_prev = torch.zeros((seq_num, model.x_dim, 1), device=device)
            min_val = -1.0  # 最小值
            max_val = 1.0  # 最大值
            x_prev = (max_val - min_val) * torch.rand((seq_num, model.x_dim, 1), device=device) + min_val
            with torch.no_grad():
                for i in range(seq_len):
                    # 状态值：
                    xt = model.f(x_prev)
                    x_mean = torch.zeros(seq_num, model.x_dim)
                    distrib = MultivariateNormal(loc=x_mean, covariance_matrix=cov_q)
                    eq = distrib.rsample().view(seq_num, model.x_dim, 1).to(device)
                    xt = torch.add(xt, eq)
                    # 观测值：
                    yt = model.H.matmul(xt)
                    y_mean = torch.zeros(seq_num, model.y_dim)
                    distrib = MultivariateNormal(loc=y_mean, covariance_matrix=cov_r)
                    er = distrib.rsample().view(seq_num, model.y_dim, 1).to(device)
                    yt = torch.add(yt, er)
                    x_prev = xt.clone()
                    state_mtx[:, :, i] = torch.squeeze(xt, 2)
                    obs_mtx[:, :, i] = torch.squeeze(yt, 2)

                state[j] = state_mtx
                obs[j] = obs_mtx
                j += 1
    torch.save(state, '../../MAML_data/linear/plot_data/state.pt')
    torch.save(obs, '../../MAML_data/linear/plot_data/obs.pt')
else:
    state = torch.load('../../MAML_data/linear/plot_data/state.pt')
    obs = torch.load('../../MAML_data/linear/plot_data/obs.pt')

for m in range(len(noise_q)):
    # 生成训练数据
    selected_task_num = np.random.choice(seq_num, args.task_num_train + args.task_num_test, False)

    state_spt = state[m*len(v):(m+1)*len(v), selected_task_num[:args.task_num_train], :, :].to(device)
    obs_spt = obs[m*len(v):(m+1)*len(v), selected_task_num[:args.task_num_train], :, :].to(device)
    state_qry = state[m*len(v):(m+1)*len(v), selected_task_num[args.task_num_train:], :, :].to(device)
    obs_qry = obs[m*len(v):(m+1)*len(v), selected_task_num[args.task_num_train:], :, :].to(device)

    losses_dB_KF = []
    i = 0
    for state_qry_one, obs_qry_one in zip(state_qry, obs_qry):
        # 计算当前时刻噪声
        q2 = noise_q[0]
        r2 = q2 * 10 ** (-v[i] / 10)
        cov_q = q2 * torch.eye(model.x_dim)
        cov_r = r2 * torch.eye(model.y_dim)
        losses_dB_KF_test = my_filter.EKF(state_qry_one, obs_qry_one, cov_q, cov_r)
        losses_dB_KF.append(losses_dB_KF_test.item())
        i = i + 1

    print('q2= ' + str(noise_q[m]) + ' KF loss(dB): ' + str(losses_dB_KF))

    if Train:
        ##########################
        # unsupervised KalmanNet #
        ##########################
        print("traning KalmanNet in unsupervised way")
        for i, (state_spt_one, obs_spt_one) in enumerate(zip(state, obs)):
            # 随机生成网络参数
            task_model = Learner(model.x_dim, model.y_dim, args, is_linear_net=True).to(device)
            task_model.load_state_dict(random_net.state_dict())
            task_model.initialize_hidden(is_train=True)
            inner_optimizer = optim.Adam(task_model.parameters(), lr=args.update_lr)
            # my_filter.train_net.load_state_dict(random_net.state_dict())

            for k in range(args.epoch):
                _ = my_filter.compute_x_post(state_spt_one, obs_spt_one, task_net=task_model)
                loss = nn.MSELoss()(my_filter.temp_y_predict_history[:, :, 1:-1], my_filter.temp_batch_y[:, :, 2:])
                inner_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), 10)
                inner_optimizer.step()
                if k % 10 == 0:
                    print(str(i) + 'th data train' + ' train num:' + str(k) + ' loss(dB): ' + str(10 * torch.log10(loss)))

            torch.save(task_model, '../../MAML_model/linear/model_trajectory/unsupervised kalmanNet/q=0.15_v=' + str(v[i]) + '.pt')

        ##########################
        ######### DANSE ##########
        ##########################
        print("traning DANSE in unsupervised way")
        data_idx = 0
        data_num = obs.shape[1]
        for i, (state_spt_one, obs_spt_one) in enumerate(zip(state, obs)):
            obs_spt_one = obs_spt_one.permute(0, 2, 1)
            state_spt_one = state_spt_one.permute(0, 2, 1)
            danse_filter.danse_net.load_state_dict(danse_dict)
            r2 = noise_q[0] * 10 ** (-v[i] / 10)
            cov_r = r2 * torch.eye(model.y_dim, device=device)
            for k in range(args.epoch):
                if data_idx + args.batch_size >= data_num:
                    data_idx = 0
                    shuffle_idx = torch.randperm(obs_spt_one.shape[0])
                    obs_spt_one = obs_spt_one[shuffle_idx]
                    state_spt_one = state_spt_one[shuffle_idx]
                batch_y = obs_spt_one[data_idx:data_idx + args.batch_size]
                batch_x = state_spt_one[data_idx:data_idx + args.batch_size]
                loss = -danse_filter.DANSE_filtering(batch_y, cov_r)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(danse_filter.danse_net.parameters(), 10)
                danse_filter.optimizer.step()
                data_idx += args.batch_size
                if k % 10 == 0:
                    with torch.no_grad():
                        te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = \
                            danse_filter.compute_predictions(batch_y, cov_r)
                        log_pY_test_batch = -danse_filter.DANSE_filtering(batch_y, cov_r)

                        loss = loss_fn(batch_x, te_mu_X_filtered_batch)
                    print(str(i) + 'th data train' + ' train num:' + str(k) + ' loss(dB): ' + str(10 * torch.log10(loss)))

            torch.save(danse_filter.danse_net, '../../MAML_model/linear/model_trajectory/danse/q=0.15_v=' + str(v[i]) + '.pt')

        ##########################
        ####### KalmanNet ########
        ##########################
        print("training KalmanNet in supervised way")
        for i, (state_spt_one, obs_spt_one) in enumerate(zip(state, obs)):
            # 随机生成网络参数
            my_filter.train_net.load_state_dict(random_net.state_dict())

            for k in range(args.epoch):
                loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
                my_filter.train_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 10)
                my_filter.train_optim.step()
                if k % 10 == 0:
                    print(str(i) + 'th data train' + ' train num:' + str(k) + ' loss(dB): ' + str(10 * torch.log10(loss)))

            torch.save(my_filter.train_net, '../../MAML_model/linear/model_trajectory/KalmanNet/q=0.15_v=' + str(v[i]) + '.pt')

    losses_dB_last = []
    losses_dB = []
    losses_dB_8 = []
    losses_dB_4 = []
    losses_dB_random = []
    losses_dB_random_last = []
    losses_dB_random_8 = []
    losses_dB_random_4 = []
    losses_dB_danse = []
    losses_dB_KalmanNet = []
    losses_dB_unsupervised_KalmanNet = []
    losses_dB_pre_trained_KalmanNet = []
    losses_dB_pre_trained_KalmanNet_last = []
    losses_dB_pre_trained_DANSE = []
    losses_dB_pre_trained_DANSE_last = []

    for i, (state_spt_one, obs_spt_one, state_qry_one, obs_qry_one) in enumerate(zip(state_spt, obs_spt, state_qry, obs_qry)):

        # 使用MAML所训练得到的初始化网络
        my_filter.train_net.load_state_dict(my_dict)

        for k in range(args.update_step_test):

            loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
            my_filter.train_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 5)
            my_filter.train_optim.step()

            with torch.no_grad():
                loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
                losses_dB.append((10 * torch.log10(loss_q)).item())

        losses_dB_last.append(losses_dB[-1])
        losses_dB_8.append(losses_dB[-9])
        losses_dB_4.append(losses_dB[-13])

        # # 随机生成网络参数
        # my_filter.train_net.load_state_dict(random_net.state_dict())
        #
        # for k in range(args.update_step_test):
        #
        #     loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
        #     my_filter.train_optim.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 5)
        #     my_filter.train_optim.step()
        #
        #     with torch.no_grad():
        #         loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
        #         losses_dB_random.append((10 * torch.log10(loss_q)).item())
        #
        # losses_dB_random_last.append(losses_dB_random[-1])
        # losses_dB_random_8.append(losses_dB_random[-9])
        # losses_dB_random_4.append(losses_dB_random[-13])

        # 预训练 KalmanNet
        my_filter.train_net.load_state_dict(pretrained_net.state_dict())

        for k in range(args.update_step_test):
            loss = my_filter.compute_x_post(state_spt_one, obs_spt_one)
            my_filter.train_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_filter.train_net.parameters(), 5)
            my_filter.train_optim.step()

            with torch.no_grad():
                loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
                losses_dB_pre_trained_KalmanNet.append((10 * torch.log10(loss_q)).item())

        losses_dB_pre_trained_KalmanNet_last.append(losses_dB_pre_trained_KalmanNet[-1])

        # # pre-training DANSE
        # danse_dict = torch.load('../../MAML_model/linear/model_trajectory/pre_training_DANSE.pt')
        # danse_filter.danse_net.load_state_dict(danse_dict)
        # obs_spt_danse = obs_spt_one.permute(0, 2, 1)
        # obs_qry_danse = obs_qry_one.permute(0, 2, 1)
        # state_qry_danse = state_qry_one.permute(0, 2, 1)
        # r2 = noise_q[0] * 10 ** (-v[i] / 10)
        # cov_r = r2 * torch.eye(model.y_dim, device=device)
        # data_idx = 0
        # for k in range(args.update_step_test):
        #     if data_idx + args.batch_size >= obs_spt_danse.shape[0]:
        #         data_idx = 0
        #         shuffle_idx = torch.randperm(obs_spt_danse.shape[0])
        #         obs_spt_danse = obs_spt_danse[shuffle_idx]
        #     batch_y = obs_spt_danse[data_idx:data_idx + args.batch_size]
        #     loss = -danse_filter.DANSE_filtering(batch_y, cov_r)
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(danse_filter.danse_net.parameters(), 10)
        #     danse_filter.optimizer.step()
        #     data_idx += args.batch_size
        #     with torch.no_grad():
        #         te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = \
        #             danse_filter.compute_predictions(obs_qry_danse, cov_r)
        #         log_pY_test_batch = -danse_filter.DANSE_filtering(obs_qry_danse, cov_r)
        #         loss = loss_fn(state_qry_danse, te_mu_X_filtered_batch)
        #         losses_dB_pre_trained_DANSE.append((10 * torch.log10(loss)).item())
        #
        # losses_dB_pre_trained_DANSE_last.append(losses_dB_pre_trained_DANSE[-1])
        #
        # # fully train DANSE
        # temp_net = torch.load('../../MAML_model/linear/model_trajectory/danse/q=0.15_v=' + str(v[i]) + '.pt')
        # danse_filter.danse_net.load_state_dict(temp_net.state_dict())
        # with torch.no_grad():
        #
        #     r2 = noise_q[0] * 10 ** (-v[i] / 10)
        #     cov_r = (r2 * torch.eye(model.y_dim)).to(device)
        #     obs_qry_danse = obs_qry_one.permute(0, 2, 1)
        #     state_qry_danse = state_qry_one.permute(0, 2, 1)
        #     te_mu_X_predictions_batch, te_var_X_predictions_batch, te_mu_X_filtered_batch, te_var_X_filtered_batch = \
        #         danse_filter.compute_predictions(obs_qry_danse, cov_r)
        #     log_pY_test_batch = -danse_filter.DANSE_filtering(obs_qry_danse, cov_r)
        #
        #     loss = loss_fn(state_qry_danse, te_mu_X_filtered_batch)
        #     losses_dB_danse.append((10 * torch.log10(loss)).item())

        # # fully train KalmanNet
        # temp_net = torch.load('../../MAML_model/linear/model_trajectory/KalmanNet/q=0.15_v=' + str(v[i]) + '.pt')
        # my_filter.train_net.load_state_dict(temp_net.state_dict())
        # with torch.no_grad():
        #     loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
        #     losses_dB_KalmanNet.append((10 * torch.log10(loss_q)).item())
        #
        # # fully train unsupervised KalmanNet
        # temp_net = torch.load('../../MAML_model/linear/model_trajectory/unsupervised kalmanNet/q=0.15_v=' + str(v[i]) + '.pt')
        # my_filter.train_net.load_state_dict(temp_net.state_dict())
        # with torch.no_grad():
        #     loss_q = my_filter.compute_x_post_qry(state_qry_one, obs_qry_one)
        #     losses_dB_unsupervised_KalmanNet.append((10 * torch.log10(loss_q)).item())

    np.save('../../result/linear/plot_loss/MAML-KalmanNet.npy', np.array(losses_dB))
    # np.save('../../result/linear/plot_loss/DANSE.npy', np.array(losses_dB_danse))
    # np.save('../../result/linear/plot_loss/KalmanNet.npy', np.array(losses_dB_KalmanNet))
    # np.save('../../result/linear/plot_loss/Unsupervised KalmanNet.npy', np.array(losses_dB_unsupervised_KalmanNet))
    np.save('../../result/linear/plot_loss/pre_training-KalmanNet.npy', np.array(losses_dB_pre_trained_KalmanNet))
    # np.save('../../result/linear/plot_loss/pre_training-DANSE.npy', np.array(losses_dB_pre_trained_DANSE))

    AKNet = [-4.1327, -7.0812, -10.4141, -13.8470, -17.9486, -21.8434, -27.0404]
    AKNet_inference = [-4.1695, -7.0780, -10.4440, -13.8892, -17.3497, -19.7097, -22.0085]

    print('q2= ' + str(noise_q[m]) + ' Final update loss: ' + str(losses_dB_last))
    print('q2= ' + str(noise_q[m]) + ' 8 times update loss: ' + str(losses_dB_8))
    print('q2= ' + str(noise_q[m]) + ' 4 times update loss: ' + str(losses_dB_4))
    # print('q2= ' + str(noise_q[m]) + ' Random init final update loss: ' + str(losses_dB_random_last))
    # print('q2= ' + str(noise_q[m]) + ' Random init 8 times update loss: ' + str(losses_dB_random_8))
    # print('q2= ' + str(noise_q[m]) + ' Random init 4 times update loss: ' + str(losses_dB_random_4))
    # print('q2= ' + str(noise_q[m]) + ' DANSE loss: ' + str(losses_dB_danse))
    # print('q2= ' + str(noise_q[m]) + ' KalmanNet loss: ' + str(losses_dB_KalmanNet))
    # print('q2= ' + str(noise_q[m]) + ' Unsupervised KalmanNet loss: ' + str(losses_dB_unsupervised_KalmanNet))
    print('q2= ' + str(noise_q[m]) + ' pre-trained KalmanNet loss: ' + str(losses_dB_pre_trained_KalmanNet_last))
    print('q2= ' + str(noise_q[m]) + ' AKNet: ' + str(AKNet))
    print('q2= ' + str(noise_q[m]) + ' AKNet_inference: ' + str(AKNet_inference))
    # print('q2= ' + str(noise_q[m]) + ' pre-trained DANSE loss: ' + str(losses_dB_pre_trained_DANSE_last))

    plt.plot(v, losses_dB_KF, color='red', label='KF', linestyle='--', marker='^', markerfacecolor='none',
             markeredgecolor='black', markersize=10)
    # plt.plot(v, losses_dB_danse, label='Fully trained DANSE', marker='D', color='#2CA02C')
    # plt.plot(v, losses_dB_KalmanNet, label='Fully trained KalmanNet', marker='p', color='#BE0E98')
    # plt.plot(v, losses_dB_unsupervised_KalmanNet, label='Fully trained unsupervised KalmanNet', marker='D', color='#9467BD')
    plt.plot(v, AKNet, label='Fully trained AKNet', marker='p', color='#C0440C')
    plt.plot(v, AKNet_inference, label='Partially trained AKNet', marker='D', color='#94D96D')
    plt.plot(v, losses_dB_pre_trained_KalmanNet_last, label='Pre-trained KalmanNet (train rounds = 16)', marker='o', color='#C47AB1')
    # plt.plot(v, losses_dB_pre_trained_DANSE_last, label='Pre-trained DANSE (train rounds = 16)', marker='h', color='#43D6E5')
    plt.plot(v, losses_dB_8, label='MAML-KalmanNet (train rounds = 8)', marker='H', color='#FF7F01')
    plt.plot(v, losses_dB_last, label='MAML-KalmanNet (train rounds = 16)', marker='^', color='#1F77B4')

plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体
fontsize = 12
plt.xlabel(r'$V [dB] = 10lg\dfrac{q2}{r2}$', fontsize=fontsize, fontweight='bold')
plt.ylabel('MSE [dB]', fontsize=fontsize)
plt.legend(fontsize=fontsize, loc=3, prop={'size': 10})
plt.grid()
plt.tight_layout()

# plt.savefig('E:/Is_shanli/实验结果/不同高斯噪声大小/Synthetic/compare with fully train net_lr=0.018_-10-20/with pre-trained KalmanNet.eps', format='eps')
plt.show()
