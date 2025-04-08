#先把target dataset固定下来，以免不匹配。
import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
from pathlib import Path
import yaml
import h5py

import algo.utils as utils

import ott
import d4rl
import scipy as sp

import jax.numpy as jnp
import numpy as np
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, output_dim),
                        )
    
    def forward(self, x):
        return self.network(x)
    
class DoubleQFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)

class ValueFunc(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueFunc, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def solve_ot(
    src_data, tar_data, cost_type='cosine'
):
    src_B = src_data.shape[0]
    tgt_B = tar_data.shape[0]

    src_embs = jnp.array(src_data.reshape(src_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)
    tgt_embs = jnp.array(tar_data.reshape(tgt_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)

    if cost_type == 'euclidean':
        cost_fn = ott.geometry.costs.Euclidean()
    elif cost_type == 'cosine':
        cost_fn = ott.geometry.costs.Cosine()
    else:
        raise NotImplementedError

    scale_cost = 'max_cost'
    geom = ott.geometry.pointcloud.PointCloud(src_embs, tgt_embs, cost_fn=cost_fn, scale_cost=scale_cost)

    solver = ott.solvers.linear.sinkhorn.Sinkhorn(threshold=1e-9, max_iterations=100)
    prob = ott.problems.linear.linear_problem.LinearProblem(geom)
    sinkhorn_output = solver(prob)
    
    coupling_matrix = geom.transport_from_potentials(
        sinkhorn_output.f, sinkhorn_output.g
    )
    cost_matrix = cost_fn.all_pairs(src_embs, tgt_embs)
    ot_costs = jnp.einsum('ij,ij->i', coupling_matrix, cost_matrix)

    return -ot_costs

def filter_dataset(src_replay_buffer, tar_replay_buffer, cost_type='cosine'):
    src_num = src_replay_buffer.state.shape[0]
    srcdata = np.hstack([src_replay_buffer.state, src_replay_buffer.action, src_replay_buffer.next_state])

    tar_num = tar_replay_buffer.state.shape[0]
    tardata = np.hstack([tar_replay_buffer.state, tar_replay_buffer.action, tar_replay_buffer.next_state])

    cost_result = []

    batch_solve = jax.jit(solve_ot)

    iter_time = src_num // 10000 + 1

    for i in range(iter_time):
        current_time = time.time()
        if 10000*i >= src_num:
            break
        Gs = batch_solve(srcdata[10000*i:10000*(i+1)], tardata)

        part_res = jax.device_get(Gs)
        part_res = part_res.tolist()

        cost_result = cost_result + part_res

        print('Have completed {} transitions'.format(10000*(i+1)))
    
    cost_result = np.array(cost_result)

    return cost_result

def plot(reduced_samples):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(reduced_samples[:5000, 0], reduced_samples[:5000, 1], color='blue', label='expert', s=10)
    plt.scatter(reduced_samples[5000:, 0], reduced_samples[5000:, 1], color='red', label='random', s=10)
    plt.legend()
    
    # 添加标题和坐标轴标签
    plt.title('Src and Tar Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图形
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/src_tar.png")
    

def plot_filter(reduced_samples, src_indices, tar_indices, filter_indices):
    plt.figure(figsize=(6, 6))
    #绘制过滤掉的点
    plt.scatter(reduced_samples[filter_indices, 0], reduced_samples[filter_indices, 1], color='gray', label='filtered', s=10)

    # 绘制 positive_list 的蓝点
    plt.scatter(reduced_samples[src_indices, 0], reduced_samples[src_indices, 1], color='blue', label='expert', s=10)

    # 绘制 negative_list 的红点
    plt.scatter(reduced_samples[tar_indices, 0], reduced_samples[tar_indices, 1], color='red', label='random', s=10)

    # 添加图例
    plt.legend(fontsize=18, loc='upper left')
    
    plt.tick_params(
        axis='both',          # 同时应用于x和y轴
        which='both',         # 同时应用于主刻度和次刻度
        # bottom=True,          # 保留底部边框
        # top=False,            # 移除顶部边框
        # left=True,            # 保留左侧边框
        # right=False,          # 移除右侧边框
        labelbottom=False,    # 移除底部标签
        labelleft=False,      # 移除左侧标签
        length=0             # 设置刻度线长度为0（不显示刻度线）
    )

    # 添加标题和坐标轴标签
    # plt.title('Src and Tar Points')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')

    # 显示图形
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/src_tar_filter_dvdf.pdf")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dir", default="./limiteddatasets")
    # parser.add_argument("--env", default="halfcheetah")        
    # parser.add_argument("--tartype", default='medium', type=str)
    
    # args = parser.parse_args()

    # # load offline datasets
    # np.random.seed(100)
    
    # tar_dataset = d4rl.qlearning_dataset(gym.make("hopper-random-v2"))
    # tar_size = 5000
    
    # ind = np.random.randint(0, tar_dataset["observations"].shape[0], size=tar_size)
    
    # tar_dataset = {
    #     "observations": tar_dataset['observations'][ind],
    #     "actions": tar_dataset['actions'][ind],
    #     "next_observations": tar_dataset['next_observations'][ind],
    #     "rewards": tar_dataset['rewards'][ind],
    #     "terminals": tar_dataset['terminals'][ind],
    # }
    
    
    # src_dataset_path = f"{str(Path(__file__).parent.absolute())}/datasets/hopper-kinematic-expert.hdf5"
    # data_dict = {}
    # with h5py.File(src_dataset_path, 'r') as dataset_file:
    #     for k in tqdm(get_keys(dataset_file), desc="load datafile"):
    #         try:  # first try loading as an array
    #             data_dict[k] = dataset_file[k][:]
    #         except ValueError as e:  # try loading as a scalar
    #             data_dict[k] = dataset_file[k][()]
    # src_dataset = data_dict
    # src_size = 5000
    
    # ind = np.random.randint(0, src_dataset["observations"].shape[0], size=src_size)
    
    # src_dataset = {
    #     "observations": src_dataset['observations'][ind],
    #     "actions": src_dataset['actions'][ind],
    #     "next_observations": src_dataset['next_observations'][ind],
    #     "rewards": src_dataset['rewards'][ind],
    #     "terminals": src_dataset['terminals'][ind],
    # }
    
    # re_dataset = {
    #     "observations": np.concatenate([src_dataset['observations'],tar_dataset["observations"]], axis=0),
    #     "actions": np.concatenate([src_dataset['actions'],tar_dataset["actions"]], axis=0),
    #     "next_observations": np.concatenate([src_dataset['next_observations'],tar_dataset["next_observations"]], axis=0),
    #     "rewards": np.concatenate([src_dataset['rewards'],tar_dataset["rewards"]], axis=0),
    #     "terminals": np.concatenate([src_dataset['terminals'],tar_dataset["terminals"]], axis=0),
    # }
    
    # # num_samples = len(re_dataset["observations"])
    # # shuffled_indices = np.random.permutation(num_samples)

    # # re_dataset_shuffled = {
    # #     "observations": re_dataset["observations"][shuffled_indices],
    # #     "actions": re_dataset["actions"][shuffled_indices],
    # #     "next_observations": re_dataset["next_observations"][shuffled_indices],
    # #     "rewards": re_dataset["rewards"][shuffled_indices],
    # #     "terminals": re_dataset["terminals"][shuffled_indices],
    # # }

    # with h5py.File("./motivation/hopper-kinematic-random-expert.hdf5", 'w') as hfile:

    #     for k in re_dataset:
    #         hfile.create_dataset(k, data=re_dataset[k], compression='gzip')
            
    # print("save dataset done")
    
    # #生成cost

    tar_env = gym.make("hopper-random-v2")


    state_dim = tar_env.observation_space.shape[0]
    action_dim = tar_env.action_space.shape[0] 
    max_action = float(tar_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cpu")


    # src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
    # tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # # load offline datasets
    # # src_dataset = d4rl.qlearning_dataset(src_env)
    # # tar_dataset = utils.call_tar_dataset(args.env, args.tartype)
    
    # src_dataset_path = f"{str(Path(__file__).parent.absolute())}/motivation/hopper-kinematic-random-expert.hdf5"
    # data_dict = {}
    # with h5py.File(src_dataset_path, 'r') as dataset_file:
    #     for k in tqdm(get_keys(dataset_file), desc="load datafile"):
    #         try:  # first try loading as an array
    #             data_dict[k] = dataset_file[k][:]
    #         except ValueError as e:  # try loading as a scalar
    #             data_dict[k] = dataset_file[k][()]
    # src_dataset = data_dict

    # src_replay_buffer.convert_D4RL(src_dataset)
    # tar_replay_buffer.convert_D4RL(tar_dataset)

    # cost = filter_dataset(src_replay_buffer, tar_replay_buffer, 'cosine')

    # print('done')
    # replay_dataset = dict(
    #     cost           =   cost,
    # )

    # with h5py.File("./motivation/hopper-kinematic-random-expert-cost.hdf5", 'w') as hfile:

    #     for k in replay_dataset:
    #         hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
            
    # print("save costlog done")
    
    dataset_path = '/data/qzj/OTDF/motivation/' + 'hopper-kinematic-random-expert.hdf5'
    data_dict = {}
    with h5py.File(dataset_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        
    re_dataset = data_dict
    
    
    cost_path = '/data/qzj/OTDF/motivation/' + 'hopper-kinematic-random-expert-cost.hdf5'

    data_dict = {}
    with h5py.File(cost_path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
        
    dataset = data_dict
    
    
    src_Q_path = f"{str(Path(__file__).parent.absolute())}/logs/Offline/hopper-kinematic/expert/100/models/model_critic"
    src_V_path = f"{str(Path(__file__).parent.absolute())}/logs/Offline/hopper-kinematic/expert/100/models/model_value"
    

    src_Q = DoubleQFunc(state_dim, action_dim, hidden_size=256).to(device)

    # aka value
    src_V = ValueFunc(state_dim, action_dim, hidden_size=256).to(device)
    
    src_Q.load_state_dict(torch.load(src_Q_path, map_location=device))
    src_V.load_state_dict(torch.load(src_V_path, map_location=device))
    src_q1, src_q2 = src_Q(torch.FloatTensor(re_dataset['observations']).to(device), torch.FloatTensor(re_dataset['actions']).to(device))
    src_q = torch.min(src_q1, src_q2)
        
    src_adv = (src_q - src_V(torch.FloatTensor(re_dataset['observations']).to(device))).squeeze()    # [batch,]
    
    src_adv = (src_adv - torch.min(src_adv)) / (torch.max(src_adv) - torch.min(src_adv))
    
    src_adv = torch.exp((src_adv - torch.mean(src_adv)) / torch.std(src_adv))

    cost = dataset['cost']
    
    src_cost = torch.FloatTensor(cost).to(device)
    
    src_cost = torch.exp((src_cost - torch.mean(src_cost)) / torch.std(src_cost))
    
    lambda_ = 0.4
    
    _, indices = torch.topk((1-lambda_)*src_adv + lambda_*src_cost, k=2500)
    
    #_, indices = torch.topk(src_cost, k=2500)

    # filter out transitions
    src_filter_num = 2500
    
    #indices = np.argpartition(src_cost, -src_filter_num)[-src_filter_num:]
    
    # indices = np.concatenate([
    #     np.argpartition(src_cost, -src_filter_num)[-src_filter_num:],
    #     np.argpartition(src_cost, src_filter_num)[:src_filter_num],
    # ])
    
    #indices = np.argpartition(src_cost, [5000,7500])[5000:7500]
    
    filter_indices = np.setdiff1d(np.arange(re_dataset["observations"].shape[0]), indices)
    
    src_indices = [indices[i] for i in range(indices.shape[0]) if indices[i] < 5000]
    tar_indices = [indices[i] for i in range(indices.shape[0]) if indices[i] >= 5000]
    
    observations = re_dataset["observations"]
    actions = re_dataset["actions"]
    next_observations = re_dataset["next_observations"]
    vector = np.concatenate([observations, actions, next_observations], axis=-1)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    reduced_samples = tsne.fit_transform(vector)
    #plot(reduced_samples)
    plot_filter(reduced_samples, src_indices, tar_indices, filter_indices)