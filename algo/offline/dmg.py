import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os



import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.distributions.transforms import Transform

def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


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
    
    
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        
        return mean * self.max_action
    

    
class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)
    

class ValueCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(ValueCritic, self).__init__()
        self.network = MLPNetwork(state_dim, 1, hidden_size)

    def forward(self, state):
        return self.network(state)


class DMG(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        replay_buffer,
        device,
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        antmaze=True,
        expectile=0.9,
        temp = 10.0,
        lam=0.25,
        lam_end=0.5,
        nu = 0.5,
        nu_end = 0.005,
    ):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.critic_target = copy.deepcopy(self.critic)
        self.value_critic = ValueCritic(state_dim, action_dim).to(device)
        self.value_critic_optimizer = torch.optim.Adam(self.value_critic.parameters(), lr=3e-4)
        
        self.replay_buffer = replay_buffer
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6)/self.policy_freq))
        self.lam = lam
        self.lam_end = lam_end
        self.expectile = expectile
        self.temp = temp
        self.nu = nu
        self.nu_end = nu_end
        self.max_weight = 100 if antmaze else 3
        self.total_it = 0
        self.decay_rate = 1
        self.exp_decay = 0.99


    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state).cpu().data.numpy().flatten()
            self.actor.train()
            return action

    def train_offline(self, batch_size=256, writer=None):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # value_critic
        with torch.no_grad():
            iql_q1, iql_q2 = self.critic_target(state, action)
            iql_q = torch.cat([iql_q1, iql_q2],dim=1)
            iql_q,_ = torch.min(iql_q,dim=1,keepdim=True)
        iql_v = self.value_critic(state)
        value_loss = expectile_loss(iql_q - iql_v, self.expectile).mean()
        self.value_critic_optimizer.zero_grad()
        value_loss.backward()
        self.value_critic_optimizer.step()

        # critic
        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        # Compute the target Q value
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q_pi = torch.cat([target_Q1, target_Q2],dim=1)
            target_Q_pi,_ = torch.min(target_Q_pi,dim=1,keepdim=True)
            target_Q_iql = self.value_critic(next_state)
            target_Q = reward + not_done * self.discount * (self.lam * target_Q_pi + (1-self.lam) * target_Q_iql)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss =  F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        if self.total_it % 10000 == 0:
            with torch.no_grad():
       
                curr_Q = torch.cat([current_Q1, current_Q2],dim=1)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            with torch.no_grad():
                awr_v = self.value_critic(state)
                awr_q1, awr_q2 = self.critic_target(state, action)
                awr_q = torch.minimum(awr_q1, awr_q2)
                exp_a = torch.exp((awr_q - awr_v) * self.temp)
                exp_a = torch.clamp(exp_a, max=self.max_weight).detach()

            v1,v2 = self.critic(state, pi)
            v = torch.cat([v1,v2], dim=1)
            vmin,_ = torch.min(v, dim=1)
            lmbda = 1.0 / vmin.abs().mean().detach() # follow TD3BC
            q_loss = -lmbda * vmin.mean()

            awr_loss = (exp_a * ((pi - action)**2)).mean()
            actor_loss = q_loss + self.nu * awr_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_lr_schedule.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.value_critic.state_dict(), filename + "_value")
        torch.save(self.value_critic_optimizer.state_dict(), filename + "_value_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.actor_lr_schedule.state_dict(), filename + "_actor_lr_scheduler")
