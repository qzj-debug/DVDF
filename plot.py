import matplotlib.pyplot as plt
import gym
import d4rl
from tqdm import tqdm
import h5py
from pathlib import Path
import numpy as np
import torch

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def plot(src_samples, tar_samples):
    plt.figure(figsize=(8, 6))

    # 绘制 positive_list 的蓝点
    plt.scatter(src_samples[:, 0], src_samples[:, 1], color='blue', label='Src', s=10)

    # 绘制 negative_list 的红点
    plt.scatter(tar_samples[:, 0], tar_samples[:, 1], color='red', label='Tar', s=10)

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Src and Tar Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图形
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/src_tar.png")
    
def plot_filter(src_samples, tar_samples, src_selected_ind, src_filtered_ind, tar_selected_ind, tar_filtered_ind):
    plt.figure(figsize=(8, 6))
    
    # 绘制过滤掉的点
    plt.scatter(src_samples[src_filtered_ind, 0], src_samples[src_filtered_ind, 1], color='gray', label='src_filter', s=10)
    plt.scatter(tar_samples[tar_filtered_ind, 0], tar_samples[tar_filtered_ind, 1], color='gray', label='Tar_filter', s=10)

    # 绘制 positive_list 的蓝点
    plt.scatter(src_samples[src_selected_ind, 0], src_samples[src_selected_ind, 1], color='blue', label='Src', s=10)

    # 绘制 negative_list 的红点
    plt.scatter(tar_samples[tar_selected_ind, 0], tar_samples[tar_selected_ind, 1], color='red', label='Tar', s=10)


    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Src and Tar Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示图形
    plt.savefig(f"{str(Path(__file__).parent.absolute())}/imgs/src_tar_filter.png")
    
    

env = "hopper-kinematic"
srctype = "expert"
src_dataset_path = f"{str(Path(__file__).parent.absolute())}/dataset/source/{env}-{srctype}.hdf5"
data_dict = {}
with h5py.File(src_dataset_path, 'r') as dataset_file:
    for k in tqdm(get_keys(dataset_file), desc="load datafile"):
        try:  # first try loading as an array
            data_dict[k] = dataset_file[k][:]
        except ValueError as e:  # try loading as a scalar
            data_dict[k] = dataset_file[k][()]
            
src_dataset = data_dict
tar_dataset = d4rl.qlearning_dataset(gym.make("hopper-random-v2"))

src_size = 5000
tar_size = 5000

seed = 100
np.random.seed(seed)

src_ind = np.random.randint(0, src_dataset["observations"].shape[0], size=src_size)
tar_ind = np.random.randint(0, tar_dataset["observations"].shape[0], size=tar_size)

src_dataset = {
    "observations": src_dataset['observations'][src_ind],
    "actions": src_dataset['actions'][src_ind],
    "next_observations": src_dataset['next_observations'][src_ind],
    "rewards": src_dataset['rewards'][src_ind],
    "terminals": src_dataset['terminals'][src_ind],
}

tar_dataset = {
    "observations": tar_dataset['observations'][tar_ind],
    "actions": tar_dataset['actions'][tar_ind],
    "next_observations": tar_dataset['next_observations'][tar_ind],
    "rewards": tar_dataset['rewards'][tar_ind],
    "terminals": tar_dataset['terminals'][tar_ind],
}


observations = np.concatenate([src_dataset["observations"], tar_dataset["observations"]], axis=0)
actions = np.concatenate([src_dataset["actions"], tar_dataset["actions"]], axis=0)
next_observations = np.concatenate([src_dataset["next_observations"], tar_dataset["next_observations"]], axis=0)

vector = np.concatenate([observations, actions, next_observations], axis=-1)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
reduced_samples = tsne.fit_transform(vector)
reduced_src_samples = reduced_samples[:src_size]
reduced_tar_samples = reduced_samples[src_size:]

#plot(reduced_src_samples, reduced_tar_samples)

src_select_ratio = 0.05

batch_size = int(0.25 * (src_size + tar_size))

src_select_size = int(batch_size * src_select_ratio)
tar_select_size = batch_size - src_select_size

src_select_ind = np.random.randint(0, src_dataset["observations"].shape[0], size=src_select_size)
tar_select_ind = np.random.randint(0, tar_dataset["observations"].shape[0], size=tar_select_size)

src_filter_ind = np.setdiff1d(np.arange(src_dataset["observations"].shape[0]), src_select_ind)
tar_filter_ind = np.setdiff1d(np.arange(tar_dataset["observations"].shape[0]), tar_select_ind)

plot_filter(reduced_src_samples, reduced_tar_samples, src_select_ind, src_filter_ind, tar_select_ind, tar_filter_ind)

#按数据值采样
# values = src_dataset["observations"][:, 0]  # 取第0列的值
# probabilities = (values - values.min()) + 1e-10  # 避免负值，加一个小常数
# probabilities /= probabilities.sum()            # 归一化

# src_select_ind = np.random.choice(
#     np.arange(len(values)), 
#     size=src_select_size, 
#     p=probabilities
# )