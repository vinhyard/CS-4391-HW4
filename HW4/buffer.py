"""
HW4 — Task 1: Replay buffer and environment interaction.

Complete the four TODO items below before moving on to vpg.py.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import cv2 as cv

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
      return mu, sigma

policy_network = PolicyNet()
#==========================================

class Buffer:
    """Experience replay buffer storing one-step transitions.

    Use-contract:
        add(state, action, reward, done)             — push one transition
        calc_reward_to_go(gamma)                     — fill self.ret_to_go
        sample(batch_size) -> tuple of numpy arrays  — draw a mini-batch
    """

    def __init__(self, sdim, adim, size, sdtype=np.float32, adtype=np.float32, ep_len=200,frame_count=3):
        self.states    = np.zeros((size, sdim), dtype=sdtype)
        self.img_states    = np.zeros((size,frame_count,63,63), dtype=sdtype)
        self.actions   = np.zeros((size, adim), dtype=adtype)
        self.rewards   = np.zeros((size, 1),    dtype=np.float32)
        self.ret_to_go = np.zeros((size, 1),    dtype=np.float32)
        self.dones     = np.zeros((size, 1),    dtype=bool)
        self.i     = 0
        self.size  = size
        self.max_i = size
        self.ep_len = ep_len

    def add(self, state, action, reward, done,img_state=9):


        self.states[self.i] = state
        # self.img_states[self.i] = img_state
        
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
            self.img_states[idxs],
            idxs
        )

    def calc_reward_to_go(self, gamma=0.975):
        reward_to_go = 0.0
        k = 0
        for k in reversed(range(self.max_i)):
            if self.dones[k, 0]:
                reward_to_go = 0.0
            reward_to_go = self.rewards[k, 0] + gamma * reward_to_go
            self.ret_to_go[k, 0] = reward_to_go


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

    for i in range(size):
        action = act(agent, observation)
        next_observation, reward, term, trunc, e = env.step(action)
        
        '''
        list_observations = []
        for i in range(3):
            img = env.render()
            # processed = preprocess(img)
        # print(img1.shape)
            frame = env.step(action)
            list_observations.append(img)
        # print(len(list_observations))
        np_img_observations = np.array(list_observations)
       '''


        done = term or trunc
        buffer.add(observation, action, reward, done)
        avg_reward_list.append(reward)
        if not done:
            observation = next_observation
        else:
            observation, info = env.reset()



    buffer.calc_reward_to_go()
    return buffer, np.mean(avg_reward_list)


def act(policy, state):
    """Sample a continuous action a ~ N(mu(state), sigma) from the policy."""
    x = torch.as_tensor(state, dtype=torch.float32)
    mu, sigma = policy(x)
    action = Normal(mu, sigma).sample()
    return rescale_actions(action, -2, 2).numpy()


def rescale_actions(action, amin, amax):
    """Rescale a tanh-squashed action from (-1, 1) to the env range [amin, amax]."""
    torque_scaled = action * amax
    return torque_scaled

def preprocess(img):
    grayscale = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    gaussianBlur = cv.GaussianBlur(grayscale,ksize = (5,5),sigmaX = 0)
    # Computing gradients of x,y
    sobelx = cv.Sobel(src = gaussianBlur,ddepth=cv.CV_64F,dx=1,dy=0,ksize=3)
    sobely = cv.Sobel(src = gaussianBlur,ddepth=cv.CV_64F,dx=0,dy=1,ksize=3)
    magnitude = cv.magnitude(sobelx,sobely)

    # Binary Mask
    # binarized = cv.threshold(src=magnitude,thresh=30.0,maxval=1.0,type=cv.THRESH_BINARY)
    # print(magnitude.shape)
    x_1, x_2, y_1,y_2 = 125,375,125,375
    cropped = magnitude[x_1:x_2,y_1:y_2]
    pyramid_down1 = cv.pyrDown(cropped)
    pyramid_down2 = cv.pyrDown(pyramid_down1)
    # print(pyramid_down2.shape)

    # cv.imshow("pyramid_down",pyramid_down1)
    # cv.waitKey(1)
    processedimage = pyramid_down2
    return processedimage


if __name__ == "__main__":
    # from vpg import train_vpg
    # env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    # agent = PolicyNet()
    # buffer_replay, mean_reward = collect_data(100, env, agent)
    # print(f"Average Reward Per Step: {mean_reward}")

    #testing frame stacking code
    # WARNING: this code will display ALL stacked frames,
    # therefore a buffer of 100 size and frame count of 3 
    # will have 300 images.

    '''
    states,actions,rewards,states, dones, rtg,_,img_states,_ = buffer_replay.sample(10)
    
    for stack in img_states:
        for img in stack:
            cv.imshow("frame",img)
            cv.waitKey(1000)
        print("next stack")
    '''