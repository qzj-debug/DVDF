import gym
import d4rl
env = gym.make("hopper-medium-v2")
dataset = env.get_dataset()