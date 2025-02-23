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

from pathlib                              import Path
from algo.call_algo                       import call_algo
from dataset.call_dataset                 import call_tar_dataset
from envs.mujoco.call_mujoco_env          import call_mujoco_env
from envs.adroit.call_adroit_env          import call_adroit_env
from envs.antmaze.call_antmaze_env        import call_antmaze_env
from envs.infos                           import get_normalized_score


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="IQL", help='policy to use')
    parser.add_argument("--env", default="ant")
    parser.add_argument('--srctype', default="random", help='dataset type used in the source domain') # only useful when source domain is offline
    # support dataset type:
    # source domain: all valid datasets from D4RL
    # target domain: random, medium, medium-expert, expert
    parser.add_argument('--shift_level', default=0.1, help='the scale of the dynamics shift. Note that this value varies on different settins')
    parser.add_argument('--mode', default=0, type=int, help='the training mode, there are four types, 0: online-online, 1: offline-online, 2: online-offline, 3: offline-offline')
    parser.add_argument("--seed", default=100, type=int)
    parser.add_argument("--save_model", default=True, type=bool)        # Save model and optimizer parameters
    parser.add_argument('--tar_env_interact_interval', help='interval of interacting with target env', default=10, type=int)
    parser.add_argument('--max_step', default=int(1e6), type=int)  # the maximum gradient step for off-dynamics rl learning
    parser.add_argument('--params', default=None, help='Hyperparameters for the adopted algorithm, ought to be in JSON format')
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()  
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # we support different ways of specifying tasks, e.g., hopper-friction, hopper_friction, hopper_morph_torso_easy, hopper-morph-torso-easy
    if '_' in args.env:
        args.env = args.env.replace('_', '-')

    if 'halfcheetah' in args.env or 'hopper' in args.env or 'walker2d' in args.env or args.env.split('-')[0] == 'ant':
        domain = 'mujoco'
    elif 'pen' in args.env or 'relocate' in args.env or 'door' in args.env or 'hammer' in args.env:
        domain = 'adroit'
    elif 'antmaze' in args.env:
        domain = 'antmaze'
    else:
        raise NotImplementedError
    print(domain)

    call_env = {
        'mujoco': call_mujoco_env,
        'adroit': call_adroit_env,
        'antmaze': call_antmaze_env,
    }
    
    if domain == 'antmaze':
        src_env_name = args.env
        src_env_name_config = args.env
    elif domain == 'adroit':
        src_env_name = args.env
        src_env_name_config = args.env.split('-')[0]
    else:
        src_env_name = args.env.split('-')[0]
        src_env_name_config = src_env_name
        
    if domain == 'antmaze':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    elif domain == 'adroit':
        src_env_name = src_env_name.split('-')[0]
        src_env_name += '-' + args.srctype + '-v0'
    else:
        src_env_name += '-' + args.srctype + '-v2'
    src_env = None
    src_eval_env = gym.make(src_env_name)
    src_eval_env.seed(args.seed)
    
    policy_config_name = 'igdf'

    # load pre-defined hyperparameter config for training
    with open(f"{str(Path(__file__).parent.absolute())}/config/{domain}/{policy_config_name}/{src_env_name_config}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if args.params is not None:
        override_params = json.loads(args.params)
        config.update(override_params)
        print('The following parameters are updated to:', args.params)

    
    print("------------------------------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("------------------------------------------------------------")
    
    #outdir = args.dir + '/' + args.policy + '/' + args.env + '-' + args.srctype + '-' + str(args.seed)
    outdir = args.dir + '/' + 'offline' + '/' + args.env + '-' + args.srctype + '-' + str(args.seed)
    
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))

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
    
    
    ## write logs to record training parameters
    with open(outdir + '/log.txt','w') as f:
        f.write('\n Policy: {}; Dataset: {}, seed: {}'.format(args.policy, args.env + '-' + args.srctype, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    src_replay_buffer.convert_D4RL(d4rl.qlearning_dataset(src_eval_env))
    if 'antmaze' in args.env:
        src_replay_buffer.reward -= 1.0

    eval_cnt = 0
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_cnt += 1

    # offline training
    for t in range(int(config['max_step'])):
        policy.train(src_replay_buffer, config['batch_size'], writer=None)

        if (t + 1) % config['eval_freq'] == 0:
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            #writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            print(f"Step: {t}  Score: {src_eval_env.get_normalized_score(src_eval_return)*100}")
            
            with open(outdir + '/return.txt', 'a') as f:
                f.write(f"{t}  {src_eval_env.get_normalized_score(src_eval_return)*100} \n")
                
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
