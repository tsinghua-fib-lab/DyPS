import os

os.environ['MKL_NUM_THREADS'] = '1'
import argparse
from copyreg import pickle
import os.path as osp
import sys

sys.path.append('../')
from simulator.envs import *
from tools.create_envs import *
from algo.PPO_noid import *
import torch
import pickle
import time

from torch.utils.tensorboard import SummaryWriter
import setproctitle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
setproctitle.setproctitle("didi@ft")


def get_parameter():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # parameter
    args.MAX_ITER = 6000
    args.TEST_ITER = 1
    args.TEST_SEED = 10000

    args.resume_iter = 0
    args.device = 'cpu'
    args.neighbor_dispatch = False  # 是否考虑相邻网格接单
    args.onoff_driver = False  # 不考虑车辆的随机下线
    # args.log_name='M2_a0.01_reward2_t2_gamma0_value_noprice_noentropy'
    # args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    # args.log_name='debug'

    args.dispatch_interval = 10  # 决策间隔/min
    args.speed = args.dispatch_interval
    args.wait_time = args.dispatch_interval
    args.TIME_LEN = int(1440 // args.dispatch_interval)  # 一天的总决策次数
    args.grid_num = 100  # 网格数量，决定了数据集
    args.driver_num = 6000  # 改变初始化司机数量
    args.city_time_start = 0  # 一天的开始时间，时间的总长度是TIME_LEN

    args.batch_size = int(1e3)
    args.actor_lr = 1e-3
    args.critic_lr = 1e-3
    args.train_actor_iters = 1
    args.train_critic_iters = 1
    args.batch_size = int(args.batch_size)
    args.gamma = 0.97
    args.lam = 0.99
    args.max_grad_norm = 10
    args.clip_ratio = 0.2
    args.ent_factor = 0.01
    args.adv_normal = True
    args.clip = True
    args.steps_per_epoch = 144
    args.grad_multi = 'mean'  # sum or mean
    # args.minibatch_num= int(round(args.steps_per_epoch*args.grid_num/args.batch_size))
    args.minibatch_num = 5
    args.parallel_episode = 5
    args.parallel_way = 'mix'  # mix, mean
    args.parallel_queue = True
    args.return_scale = False
    args.use_orthogonal = True
    args.use_value_clip = True
    args.use_valuenorm = False
    args.use_huberloss = True
    args.use_lr_anneal = False
    args.use_GAEreturn = True
    args.use_rnn = False
    args.use_GAT = False
    args.use_dropout = False
    args.use_auxi = False
    args.auxi_effi = 0.1
    args.use_fake_auxi = 0
    args.use_regularize = ['None', 'L1', 'L2', 'L1state', 'L2state'][0]
    args.regularize_alpha = 1e-1

    args.use_neighbor_state = False  # 表示使用固定的多少阶邻居的信息作为状态
    args.adj_rank = 2
    args.merge_method = 'cat'  # ['cat','res']
    args.actor_centralize = False
    args.critic_centralize = False

    args.order_value = False
    args.new_order_entropy = True
    args.update_value = False
    args.order_grid = True
    args.reward_scale = 5
    args.memory_size = int(args.TIME_LEN * args.parallel_episode)
    args.FM = False
    args.remove_fake_order = False
    args.team_reward_factor = 5
    args.team_rank = 0
    args.full_share = True
    args.global_share = False
    args.ORR_reward = False
    args.ORR_reward_effi = 1
    args.only_ORR = False
    args.fix_phi = False
    args.phi = [0.025088, 0.087006, 0.184027, 0.236112, 0.218559, 0.249208]

    log_name_dict = {
        'OD': args.grid_num,
        'Batch': args.batch_size,
        # 'Advnorm': '' if args.adv_normal else 'NO',
        # 'Grad': args.grad_multi,
        'Gamma': args.gamma,
        'Lambda': args.lam,
        'Iter': args.train_actor_iters,
        'Ir': args.actor_lr,
        'Step': args.steps_per_epoch,
        # 'Clipnew': args.clip_ratio if args.clip else 'NO',
        'Ent': args.ent_factor,
        'Minibatch': args.minibatch_num,
        'Parallel': str(args.parallel_episode) + args.parallel_way,
        # 'Rscale':args.reward_scale,
        'value': '' if args.order_value else 'NO',
        'queue': '' if args.parallel_queue else 'NO',
        # 'TeamR': 'share' if args.full_share else args.team_reward_factor,
        'TeamRank': 'global' if args.global_share else args.team_rank,
        'ORR': args.ORR_reward_effi if args.ORR_reward else 'NO',
        'Actor': 'Cen' if args.actor_centralize else 'Decen',
        'Critic': 'Cen' if args.critic_centralize else 'Decen',
        'Auxi': args.auxi_effi if args.use_auxi else 'No',
        'FakeNewAuxi': args.use_fake_auxi
    }
    args.log_name = ''
    for k, v in log_name_dict.items():
        args.log_name += k + str(v) + '_'
    # args.log_name+='seed0'
    # args.log_name+='_car50'
    if args.order_grid == False:
        args.log_name += '_RmGrid'
    if args.only_ORR:
        args.log_name += '_onlyORR'
    if args.fix_phi:
        args.log_name += '_fixPhi'
    if args.update_value:
        args.log_name += '_UpVal'
    # args.log_name+='_KLNEW'
    if args.new_order_entropy:
        args.log_name += '_NewEntropy'
    if args.use_orthogonal == True:
        args.log_name += '_OrthoInit'
    if args.use_value_clip:
        args.log_name += '_ValueClip'
    if args.use_valuenorm:
        args.log_name += '_ValueNorm'
    if args.use_huberloss:
        args.log_name += '_Huberloss'
    if args.use_lr_anneal:
        args.log_name += '_LRAnneal'
    if args.use_GAEreturn:
        args.log_name += '_GAEreturn'
    if args.use_rnn:
        args.log_name += '_GRU2'
    if args.use_GAT:
        args.log_name += '_GATnew'
    if args.use_neighbor_state:
        args.log_name += '_Statenew' + str(args.adj_rank)
    if args.use_regularize is not 'None':
        args.log_name += '_' + args.use_regularize + str(args.regularize_alpha)
    # args.log_name+= '_'+args.merge_method
    # args.log_name+='_GAE'

    # args.log_name='advnormal_gradMean_iter10_lr3e-4_step144_clipno_batchall3_parallel1_minibatch1'
    # args.log_name='debug'

    current_time = time.strftime("%Y%m%d_%H-%M")
    # log_dir = '../logs/' + "{}".format(current_time)
    log_dir = '../logs/' + 'synthetic/' + 'PPO2/' + args.log_name
    args.log_dir = log_dir
    mkdir_p(log_dir)
    print("log dir is {}".format(log_dir))

    args.writer_logs = True
    if args.writer_logs:
        args_dict = args.__dict__
        with open(log_dir + '/setting.txt', 'w') as f:
            for key, value in args_dict.items():
                f.writelines(key + ' : ' + str(value) + '\n')

    return args


def train(env, agent, writer=None, args=None, device='cpu'):
    best_gmv = 0
    best_orr = 0
    if args.return_scale:
        record_return = test(env, agent, test_iter=1, args=args, device=device) / 20
        record_return[record_return == 0] = 1
    for iteration in np.arange(args.resume_iter, args.MAX_ITER):
        t_begin = time.time()
        print('\n---- ROUND: #{} ----'.format(iteration))
        RANDOM_SEED = iteration + args.MAX_ITER
        env.reset_randomseed(RANDOM_SEED)

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []
        order_response_rates = []
        T = 0

        states_node, _, order_states, order_idx, order_feature, global_order_states = env.reset(mode='PPO2')
        state = agent.process_state(states_node, T)  # state dim= (grid_num, 119)
        state_rnn_actor = torch.zeros((1, agent.agent_num, agent.hidden_dim), dtype=torch.float)
        state_rnn_critic = torch.zeros((1, agent.agent_num, agent.hidden_dim), dtype=torch.float)
        order_states = agent.add_order_value(order_states)
        order, mask_order = agent.process_order(order_states)
        order = agent.remove_order_grid(order)
        mask_order = agent.mask_fake(order, mask_order)

        for T in np.arange(args.TIME_LEN):
            assert len(order_idx) == args.grid_num, 'dim error'
            assert len(order_states) == args.grid_num, 'dim error'
            for i in range(len(order_idx)):
                assert len(order_idx[i]) == len(order_states[i]), 'dim error'

            # t0=time.time()
            MDP.cur_time = T
            action, value, logp, mask_agent, mask_order_multi, mask_action, next_state_rnn_actor, next_state_rnn_critic, action_ids, selected_ids = agent.action(
                state, order, state_rnn_actor, state_rnn_critic, mask_order, order_idx, device, sample=False,
                random_action=False, MDP=MDP, fleet_help=args.FM)

            if args.order_value and args.update_value:
                MDP.update_value(order_states, selected_ids, env)

            # t1=time.time()
            orders = env.get_orders_by_id(action_ids)

            next_states_node, next_order_states, next_order_idx, next_order_feature = env.step(orders, generate_order=1,
                                                                                               mode='PPO2')

            # t2=time.time()

            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            entropy.append(entr_value)
            kl.append(kl_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            # store transition
            if T == args.TIME_LEN - 1:
                done = True
            else:
                done = False
            reward = torch.Tensor([node.gmv for node in env.nodes])

            if args.global_share:
                reward = torch.mean(reward, 0, keepdim=True).repeat(args.grid_num)
            else:
                if args.full_share == False:
                    if args.fix_phi:
                        team_reward = torch.zeros_like(reward)
                        for i in range(args.grid_num):
                            for rank in range(args.team_rank):
                                neighb = env.layer_neighborhood[i][rank]
                                team_reward[i] += torch.mean(reward[neighb]) * args.phi[rank + 1]
                        reward = args.phi[0] * reward + team_reward
                    else:
                        team_reward = torch.zeros_like(reward)
                        for i in range(args.grid_num):
                            for rank in range(args.team_rank):
                                neighb = env.layer_neighborhood[i][rank]
                                team_reward[i] += torch.mean(reward[neighb])
                        reward = 1 / np.sqrt(
                            1 + args.team_reward_factor ** 2) * reward + args.team_reward_factor / np.sqrt(
                            1 + args.team_reward_factor ** 2) * team_reward
                else:
                    team_reward = torch.zeros_like(reward)
                    for i in range(args.grid_num):
                        num = 1
                        team_reward[i] = reward[i]
                        for rank in range(args.team_rank):
                            neighb = env.layer_neighborhood[i][rank]
                            num += len(neighb)
                            team_reward[i] += torch.sum(reward[neighb])
                        team_reward[i] /= num
                    reward = team_reward

            if args.ORR_reward == True:
                ORR_reward = torch.zeros_like(reward)
                driver_num = torch.Tensor([node.idle_driver_num for node in env.nodes]) + 1e-5
                order_num = torch.Tensor([node.real_order_num for node in env.nodes]) + 1e-5
                driver_order = torch.stack([driver_num, order_num], dim=1)
                ORR_entropy = torch.min(driver_order, dim=1)[0] / torch.max(driver_order, dim=1)[0]
                '''
                ORR_entropy= ORR_entropy*torch.log(ORR_entropy)

                global_entropy= torch.min(torch.sum(driver_order,dim=0))/torch.max(torch.sum(driver_order,dim=0))
                global_entropy = global_entropy*torch.log(global_entropy)
                ORR_entropy= torch.abs(ORR_entropy-global_entropy)
                order_num/=torch.sum(order_num)
                driver_num/=torch.sum(driver_num)
                ORR_KL = torch.sum(order_num * torch.log(order_num / driver_num))
                '''
                for i in range(args.grid_num):
                    num = 1
                    ORR_reward[i] = ORR_entropy[i]
                    for rank in range(args.team_rank):
                        neighb = env.nodes[i].layers_neighbors_id[rank]
                        num += len(neighb)
                        ORR_reward[i] += torch.sum(ORR_entropy[neighb])
                    ORR_reward[i] /= num
                # ORR_reward= -ORR_reward*10-ORR_KL+2.5
                reward += ORR_reward * args.ORR_reward_effi
                if args.only_ORR:
                    reward = ORR_reward * args.ORR_reward_effi

            # print(0)
            if args.return_scale:
                reward /= record_return
            else:
                reward /= args.reward_scale

            next_order_states = agent.add_order_value(next_order_states)
            next_state = agent.process_state(next_states_node, T)  # state dim= (grid_num, 119)
            next_order, next_order_mask = agent.process_order(next_order_states)
            next_order = agent.remove_order_grid(next_order)
            next_order_mask = agent.mask_fake(next_order, next_order_mask)

            agent.buffer.push(state, next_state, order, action, reward[:, None], value, logp, mask_order_multi,
                              mask_action, mask_agent, state_rnn_actor.squeeze(0), state_rnn_critic.squeeze(0))

            epoch_ended = (T % args.steps_per_epoch) == (args.steps_per_epoch - 1)
            done = T == args.TIME_LEN - 1
            if done or epoch_ended:
                if done:
                    next_value = torch.zeros((agent.agent_num, 1))
                elif epoch_ended:
                    next_value, _ = agent.critic(next_state.to(device), agent.adj,
                                                 next_state_rnn_critic.to(device)).detach().cpu()
                agent.buffer.finish_path(next_value)
                # agent.update(device,writer)

            # t3=time.time()
            # print(t1-t0,t2-t0,t3-t0)

            states_node = next_states_node
            order_idx = next_order_idx
            order_states = next_order_states
            order_feature = next_order_feature
            state = next_state
            order = next_order
            mask_order = next_order_mask
            state_rnn_actor = next_state_rnn_actor
            state_rnn_critic = next_state_rnn_critic
            T += 1

        if args.parallel_queue == False:
            if (iteration + 1) % args.parallel_episode == 0:
                agent.update(device, writer)
        else:
            if (iteration + 1) >= args.parallel_episode:
                agent.update(device, writer)
                agent.buffer

        t_end = time.time()

        if np.sum(gmv) > best_gmv:
            best_gmv = np.sum(gmv)
            best_orr = order_response_rates[-1]
            agent.save_param(args.log_dir, 'Best')
        print(
            '>>> Time: [{0:<.4f}] Mean_ORR: [{1:<.4f}] GMV: [{2:<.4f}] Best_no_ORR: [{3:<.4f}] Best_no_GMV: [{4:<.4f}]'.format(
                t_end - t_begin, order_response_rates[-1], np.sum(gmv), best_orr, best_gmv))
        agent.save_param(args.log_dir, 'param')
        writer.add_scalar('train no ps ORR', order_response_rates[-1], iteration)
        writer.add_scalar('train no ps GMV', np.sum(gmv), iteration)
        # writer.add_scalar('train KL',np.mean(kl),iteration)
        # writer.add_scalar('train Suply/demand',np.mean(entropy),iteration)

        if args.order_value:
            writer.add_scalar('train value feature', np.mean(np.abs(MDP.value_iter)), iteration)
            MDP.value_iter = []
            if iteration % 10 == 0:
                MDP.save_param(args.log_dir)


def test(env, agent, test_iter=1, writer=None, args=None, device='cpu'):
    best_gmv = 0
    best_orr = 0
    record_return = torch.zeros((args.TIME_LEN, args.grid_num))
    record_driver = torch.zeros((args.TIME_LEN, args.grid_num))
    record_prob = []
    for iteration in np.arange(test_iter):
        print('\n---- ROUND: #{} ----'.format(iteration))
        RANDOM_SEED = iteration + args.MAX_ITER
        env.reset_randomseed(RANDOM_SEED)

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []
        order_response_rates = []
        T = 0

        states_node, _, order_states, order_idx, order_feature, global_order_states = env.reset(mode='PPO2')
        state = agent.process_state(states_node, T)  # state dim= (grid_num, 119)
        order, mask_order = agent.process_order(order_states)
        for T in np.arange(args.TIME_LEN):
            assert len(order_idx) == args.grid_num, 'dim error'
            assert len(order_states) == args.grid_num, 'dim error'
            for i in range(len(order_idx)):
                assert len(order_idx[i]) == len(order_states[i]), 'dim error'

            MDP.cur_time = T
            action, value, logp, mask_agent, mask_order_multi, mask_action, action_ids, full_prob, driver_num = agent.action(
                state, order, mask_order, order_idx, device, sample=False, random_action=False, MDP=MDP,
                fleet_help=args.FM, need_full_prob=True, random_fleet=False and args.FM)
            record_driver[T] = driver_num
            record_prob.append(full_prob[:, :20])

            orders = env.get_orders_by_id(action_ids)

            next_states_node, next_order_states, next_order_idx, next_order_feature = env.step(orders, generate_order=1,
                                                                                               mode='PPO2')

            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            entropy.append(entr_value)
            kl.append(kl_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            # store transition
            if T == args.TIME_LEN - 1:
                done = True
            else:
                done = False
            reward = torch.Tensor([node.gmv for node in env.nodes])
            if args.return_scale:
                reward /= record_return
            else:
                reward /= args.reward_scale
            agent.buffer.push(state, order, action, reward[:, None], value, logp, mask_order_multi, mask_action,
                              mask_agent)

            next_state = agent.process_state(next_states_node, T)  # state dim= (grid_num, 119)
            next_order, next_order_mask = agent.process_order(next_order_states)

            states_node = next_states_node
            order_idx = next_order_idx
            order_states = next_order_states
            order_feature = next_order_feature
            state = next_state
            order = next_order
            mask_order = next_order_mask
            T += 1

        print(
            '>>> Mean_ORR: [{0:<.4f}] GMV: [{1:<.4f}] Mean_KL: [{2:<.4f}] Mean_Entropy: [{3:<.4f}] Best_ORR: [{4:<.4f}] Best_GMV: [{5:<.4f}]'.format(
                order_response_rates[-1], np.sum(gmv), np.mean(kl), np.mean(entropy), best_orr, best_gmv))
        '''
        writer.add_scalar('train ORR',np.mean(order_response_rates),iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        writer.add_scalar('train KL',np.mean(kl),iteration)
        writer.add_scalar('train Suply/demand',np.mean(entropy),iteration)
        if args.order_value:
            writer.add_scalar('train value feature',np.mean(np.abs(MDP.value_iter)),iteration)
            MDP.value_iter=[]
        '''
    test_log = {
        'order': agent.buffer.order_pool[:, :144, :20, 4],
        'action': agent.buffer.action_pool[:, :144, :20],
        'reward': agent.buffer.reward_pool[:, :144],
        'advantge': agent.buffer.advantage_pool[:, :144],
        'return': agent.buffer.return_pool[:, :144],
        'driver': record_driver,
        'prob': torch.stack(record_prob, dim=1)
    }
    # with open(args.log_dir+'/'+'test_log.pkl','wb') as f:
    # pickle.dump(test_log,f)
    return torch.sum(record_return, dim=0)


if __name__ == "__main__":
    args = get_parameter()

    # if args.device == 'gpu':
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device('cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    dataset=kdd18(args)
    dataset.build_dataset(args)
    env=CityReal(dataset=dataset,args=args)
    '''
    if args.grid_num == 100:
        env, args.M, args.N, _, args.grid_num = create_OD()
    elif args.grid_num == 36:
        env, args.M, args.N, _, args.grid_num = create_OD_36()
    env.fleet_help = args.FM

    if args.writer_logs:
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    agent = PPO(env, args, device)
    MDP = MdpAgent(args.TIME_LEN, args.grid_num, args.gamma)
    if args.order_value:
        MDP.load_param('../logs/synthetic/MDP/OD+localFM/MDPsave.pkl')
        # logs/synthetic/MDP/OD+randomFM/MDP.pkl
    agent.MDP = MDP
    # agent=None
    agent.move_device(device)
    # args.log_dir='../logs/synthetic/PPO2/OD_Advnorm_Gradmean_Iter5_Ir0.0003_Step144_ClipnewNO_Ent0.0_Minibatch5_Parallel5mean_seed0'
    model_dir = '../logs/MT/synthetic/PPO2/OD_Advnorm_Gradmean_Iter1_Ir0.001_Step144_Clipnew0.2_Ent0.01_Minibatch5_Parallel5mix_Rscale5_value_queue_TeamRshare_TeamRankglobal_ORRNO_seed0_car50_KLNEW/Best.pkl'
    # agent.load_param(model_dir)
    # agent.step=args.resume_iter

    # test(env,agent, test_iter=1,args=args,device=device)
    train(env, agent, writer=writer, args=args, device=device)