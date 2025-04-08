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

import seaborn as sns

sns.set_style("white")

#plt.style.use('seaborn-white')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.size'] = 15

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

def plot_result():
    # 示例数据
    plt.figure(figsize=(6, 6))
    algos = ['IGDF', 'DVDF']
    scores = [39, 67]

    # 使用Matplotlib
    #plt.bar(algos, scores, color='skyblue', edgecolor='black')
    #plt.title('Basic Bar Chart')
    #plt.xlabel('Normalized Score')
    plt.ylabel('Normalized Score', fontsize=20)
    #plt.show()
    
    error = [[6, 7],[4,6]]
    colors = ['orange', 'cornflowerblue']
    # 绘制
    plt.bar(algos, scores, capsize=4, yerr=error, color=colors, width=1)
    # 添加标题和标签
    #plt.title('CQL')
    plt.ylim(0,100)
    #plt.xlabel('Algo')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    #plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.06)
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/performance.pdf")
    
    

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
    plt.legend(fontsize=14, loc='upper left')
    
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
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/src_tar_filter.pdf")


if __name__ == "__main__":

    plot_result()