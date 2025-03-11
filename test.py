#探究对dynamics shift的抵抗能力，用在clean环境上训练的IQL加入dynamics shift测试鲁棒性。
import numpy as np
import torch
import gym
import argparse
import os
import random
import math
import time
import copy
import yaml
import json # in case the user want to modify the hyperparameters
import d4rl # used to make offline environments for source domains
import d4rl
import algo.utils as utils
import h5py
from tqdm import tqdm
from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score

from gym.envs.mujoco.half_cheetah_v3    import  HalfCheetahEnv
from gym.envs.mujoco.ant_v3             import  AntEnv
from gym.envs.mujoco.walker2d_v3        import  Walker2dEnv
from gym.envs.mujoco.hopper_v3          import  HopperEnv

from gym.wrappers.time_limit            import  TimeLimit


def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="IQL", help='policy to use')
    parser.add_argument("--env", default="halfcheetah-kinematic-footjnt")
    parser.add_argument('--srctype', default="expert", help='dataset type used in the source domain') # only useful when source domain is offline
    parser.add_argument('--shift_level', default="hard", help='the scale of the dynamics shift. Note that this value varies on different settins')
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()  
    
    #device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")
    
    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')
    
    if "halfcheetah" in args.env:
        src_env = HalfCheetahEnv
    elif "hopper" in args.env:
        src_env = HopperEnv
    elif "walker2d" in args.env:
        src_env = Walker2dEnv
    elif "ant" in args.env:
        src_env = AntEnv
    else:
        raise NotImplementedError
    
    env_config_name = args.env.split("-")[0]
    
    src_eval_env = TimeLimit(
                    src_env(xml_file=f"{str(Path(__file__).parent.absolute())}/envs/mujoco/assets/{args.env.replace('-', '_')}_{args.shift_level}.xml",),
                    max_episode_steps=1000          
                )
    src_eval_env.seed(args.seed)
    
    ref_env_name = args.env + '-' + str(args.shift_level)
    
    
    
    policy_config_name = 'igdf'

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/mujoco/{policy_config_name}/{env_config_name}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    
    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")

    # seed all
    src_eval_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # get necessary information from both domains
    state_dim = src_eval_env.observation_space.shape[0]
    action_dim = src_eval_env.action_space.shape[0] 
    max_action = float(src_eval_env.action_space.high[0])
    min_action = -max_action
    

    config.update({
        'env_name': args.env,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'tar_env_interact_interval': int(args.tar_env_interact_interval),
        'max_step': int(args.max_step),
    })

    from algo.offline.iql import IQL
    
    algo = IQL
    policy = algo(config, device)
    
    policy.policy.load_state_dict(torch.load(f"./testlogs/IQL/percent/{env_config_name}/{args.srctype}/{args.seed}/models/model_actor", map_location=device))
    
    eval_return = eval_policy(policy, src_eval_env, eval_cnt=0)
    
    eval_normalized_score = get_normalized_score(eval_return, ref_env_name)
    
    print("eval_normalized_score:", eval_normalized_score)
