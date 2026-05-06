"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym

from Modules import NormalModule

# ========= POLICY NETWORK ==============
class PolicyNet(nn.Module):
  def __init__(self):
    super().__init__()
    # self.flatten = nn.Flatten(start_dim=-1)
    self.policy_network = nn.Sequential(
        nn.Linear(3,100), # input is state [x,y,angle]
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,10), 
        NormalModule(10,1) # output is action [mu,sigma] for Gaussian
    )
  def forward(self, x):
    mu, sigma = self.policy_network(x)
    # print(f"mu: {mu}, sigma: {sigma}")
    # ask about using NormalModule()....
    
    gaussian = Normal(mu,sigma)
    # print(f"gaussian: {gaussian}")
    torque = gaussian.sample()

    return torque

policy_network = PolicyNet()
#==========================================

class Buffer:
    """Experience replay buffer storing one-step transitions.

    Use-contract:
        add(state, action, reward, done)             — push one transition
        calc_reward_to_go(gamma)                     — fill self.ret_to_go
        sample(batch_size) -> tuple of numpy arrays  — draw a mini-batch
    """

    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32, ep_len=200):
        self.states    = np.zeros((size, sdim), dtype=sdtype)
        self.actions   = np.zeros((size, adim), dtype=adtype)
        self.rewards   = np.zeros((size, 1),    dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1),    dtype=np.float32)
        self.dones     = np.zeros((size, 1),    dtype=bool)
        self.i     = 0
        self.size  = size
        self.max_i = size
        self.ep_len = ep_len

    def add(self, state, action, reward, done):
        print(f"received state: {state}")
        print(f"received action: {action}")
        print(f"received reward: {reward}")
        print(f"received state: {state}")

        self.states[self.i] = state
        self.actions[self.i] = action
        self.rewards[self.i] = reward
        self.dones[self.i] = done

        self.i += 1
        if self.i >= self.max_i:
            self.i = 0

    def sample(self, batch_size):
        upper = max(self.max_i - 1, 1)
        idxs = np.random.randint(0, upper, size=batch_size)
        done_mask = self.dones[idxs, 0]
        idxs = np.where(done_mask, np.maximum(idxs - 1, 0), idxs)
        next_idxs = idxs + 1
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[next_idxs],
            self.dones[next_idxs],
            self.ret_to_go[idxs],
            self.ret_to_go[next_idxs],
        )

    def calc_reward_to_go(self, gamma=0.975):
        reward_to_go = 0
        k = 0
        for reward in self.rewards:
            reward_to_go = reward_to_go + ((gamma)**(k))*reward

        self.ret_to_go = reward_to_go
        print(f"reward-to-go = {self.ret_to_go}")


def collect_data(size, env, agent, title="collecting"):
    """Roll out `agent` (a policy network) in `env` for `size` steps.

    Returns:
        buffer  — a populated Buffer
        avg_rwd — average per-step reward observed during the rollout
    """
    buffer = Buffer(sdim=3,adim=1,size=size)

    avg_reward_list = []

    # Params for gym.make mode: str
    # Render mode for the Gymnaium environment
    # Options:
    # "human": see a visual window with the environment
    # "rgb_array": get image arrays
    # "None": Fast for training

    # Personal Note: rgb_array might be useful for training based off of images alone


    # OpenAI Gymnasium official Getting Started Page
    # Can be found: https://gymnasium.farama.org/introduction/basic_usage/

    # Action Space:
    # The action is a ndarray with shape (1,) representing the torque applied to free end of the pendulum.

    # Observation Space:
    # The observation is a ndarray with shape (3,) representing the x-y coordinates of the pendulum’s free end and its angular velocity.


    # Reset environment to start a new episode
    observation, info = env.reset()

    # observation: what the agent can "see"
    # info: extra debugging information (usually not needed for basic learning)

    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:

        print(f"State: {observation}")
        action = act(agent,observation) #get torque from policy network
        observation,reward,terminated,truncated,info = env.step(action)
        buffer.add(observation,action,reward,1)

        # debugging statements
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Next State: {observation}")
        avg_reward_list.append(reward)
        total_reward+=reward
        episode_over = terminated or truncated
    print(f"Episode finished! Total reward: {total_reward}")

    # debugging statements
    print(f"buffer states: {buffer.states}")
    print(f"buffer actions: {buffer.actions}")
    print(f"buffer rewards: {buffer.rewards}")
    print(f"buffer dones: {buffer.dones}")

    env.close()

    buffer.calc_reward_to_go()
    return (buffer, np.mean(avg_reward_list))


def act(policy, state):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    x_np = torch.from_numpy(state)
    torque = policy(x_np)
    torque = rescale_actions(torque,-2,2)
    return torque


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    torque_scaled = (action - amin)/(amax - amin)
    return torque_scaled


env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
agent = PolicyNet()
size = 100

buffer_replay, mean_reward = collect_data(size,env,agent)
print(f"Average Reward Per Step: {mean_reward}")