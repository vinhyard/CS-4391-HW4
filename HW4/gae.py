"""
HW4 — Task 3: Critic network and Generalized Advantage Estimation (GAE).

Depends on: buffer.py (Task 1) and vpg.py (Task 2).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor


# # ========= POLICY NETWORK ==============
# from Modules import NormalModule

# class PolicyNet(nn.Module):
#   def __init__(self):
#     super().__init__()
#     # self.flatten = nn.Flatten(start_dim=-1)
#     self.policy_network = nn.Sequential(
#         nn.Linear(3,100), # input is state [x,y,angle]
#         nn.ReLU(),
#         nn.Linear(100,100),
#         nn.ReLU(),
#         nn.Linear(100,10), 
#         NormalModule(10,1) # output is action [mu,sigma] for Gaussian
#     )
#   def forward(self, x):
#     mu, sigma = self.policy_network(x)
#     # print(f"mu: {mu}, sigma: {sigma}")
#     # ask about using NormalModule()....
    
#     gaussian = Normal(mu,sigma)
#     # print(f"gaussian: {gaussian}")
#     torque = gaussian.sample()

#     return torque

# policy_network = PolicyNet()
# #==========================================





# ---------------------------------------------------------------------------
# Shared helper (provided — do not modify)
# ---------------------------------------------------------------------------

def build_critic(state_dim, hidden_size):
    """Two-layer feed-forward critic that outputs a scalar V(s) (provided).

    Architecture:
        Linear(state_dim, hidden_size) -> ReLU
        -> Linear(hidden_size, hidden_size) -> ReLU
        -> Linear(hidden_size, 1)
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )


# ---------------------------------------------------------------------------
# Task 3 TODOs
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, next_values, dones, gamma=0.975, lam=0.95):
    """Generalized Advantage Estimation (PPO paper, Equations 11-12). https://arxiv.org/pdf/1707.06347

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

        My Implementation:
        delta_t = rewards + gamma * next_values * (1 - done_t) - values

        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

        My Implementation: 
        Isn't A approximated as  = E{r + gamma * (next_values) - values}
        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    """
    # print(dones)
    # print(type(dones))
    # print(dones.shape)
    delta_t = rewards + gamma * next_values * (1-dones) - values
    advantages = delta_t + gamma * lam * (1-dones) * (rewards + gamma*(next_values)-values)
    #                                                   r     + gamma (V'(s'))  - V(s) 
    return advantages


def reinforce_adv_signal(policy, states, actions, advantages):
    """Policy-gradient loss weighted by arbitrary advantages (e.g. GAE)."""
    # TODO: compute  -E[ A_t * log pi(a | s) ].
    loss = advantages * _log_prob(policy=policy,states=states,actions=actions)
    # print(loss.shape)
    # print(loss)
    return loss.mean()


def train_advantage_vpg(
    epochs=3,
    episodes=10,
    updates=10,
    critic_updates=80,
    learning_rate=1e-4,
    critic_lr=3e-4,
    hidden_size=32,
    layers=2,
    batch_size=10,
    gamma=0.975,
    lam=0.95,
):
    """Policy gradient with a learned V(s) baseline and GAE (Task 3).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-epoch average episodic returns
    """
    env = gym.make("Pendulum-v1")
    state_dim   = env.reset()[0].shape[0]
    action_dim  = env.action_space.sample().shape[0]
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = torch.optim.Adam(params=policy.parameters(), lr=learning_rate)
    cr_optimizer = torch.optim.Adam(params=critic.parameters(), lr=critic_lr)

    returns_per_epoch = []

    for x in range(epochs):
        print(f"{x}/{epochs} epochs")
        # --- collect experience ---
        with torch.no_grad():
            buffer, avg_rwd = collect_data(
                episodes * episode_len, env, policy, title=f"gae {x + 1}/{epochs}"
            )

        # TODO: fill buffer.ret_to_go using buffer.calc_reward_to_go(gamma).
        buffer.calc_reward_to_go(gamma=gamma)

        # --- train the critic ---
        # Regress V(s) toward the reward-to-go targets for critic_updates steps.
        for _ in range(critic_updates):
            states, actions, rewards, states, dones, rtg, _ = buffer.sample(batch_size)
            states_t = torch.as_tensor(states, dtype=torch.float32)
            rtg_t    = torch.as_tensor(rtg,    dtype=torch.float32)
            cr_optimizer.zero_grad()
            # TODO: compute mse_loss between critic(states_t) and rtg_t,
            #       then call .backward() and cr_optimizer.step().
            mse = mse_loss(critic(states_t),rtg_t).backward()
            cr_optimizer.step()

        # --- compute GAE advantages ---
        # Run the critic (no gradients) on every stored state.
        all_states = torch.as_tensor(buffer.states[: buffer.max_i], dtype=torch.float32)
        with torch.no_grad():
            values = critic(all_states).numpy()          # V(s_t)
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]                    # V(s_{t+1}), 0 at episode end

        # TODO: call compute_gae(...) to get an (N, 1) array of advantages.
        # print("------dones")
        # print(dones)
        # print(len(dones))

        # print("--------rewards")
        # print(rewards)
        # print(len(rewards))
        # print("------buffer rewards")
        # print(buffer.rewards)
        # print(len(buffer.rewards))

        # print("--------values")
        # print(values)
        # print(len(values))      

        advantages = compute_gae(rewards=buffer.rewards,values = values, next_values = next_values, dones = buffer.dones)  # TODO
        # print(f"-------advantages: {advantages}")
        advantages = np.array(advantages)
        # Normalise for training stability (provided).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # print("In training:---")
        # --- train the actor ---
        for _ in range(updates):
            idxs      = np.random.randint(0, buffer.max_i, size=batch_size)
            # print(batch_size)

            states_t  = torch.as_tensor(buffer.states[idxs],  dtype=torch.float32)
            actions_t = torch.as_tensor(buffer.actions[idxs], dtype=torch.float32)
            adv_t     = torch.as_tensor(advantages[idxs],     dtype=torch.float32)
            optimizer.zero_grad()
            # TODO: call reinforce_adv_signal(...) to get the loss,
            #       then call .backward() and optimizer.step().
            loss = reinforce_adv_signal(policy, states_t, actions_t, adv_t)
            loss.backward()
            optimizer.step()


        ep_return = avg_rwd * episode_len
        returns_per_epoch.append(ep_return)
        print(f"gae epoch {x + 1}/{epochs}: return={ep_return:.2f}")

    return policy, returns_per_epoch


if __name__ == "__main__":
    from plotting import plot_learning_curves
    from video import record_video
    from vpg import train_vpg

    # Example: compare rewards-to-go vs GAE
    policy_rtg, ret_rtg = train_vpg(epochs=200, learning_rate=3e-4)
    policy_gae, ret_gae = train_advantage_vpg(epochs=200, learning_rate=3e-4)
    plot_learning_curves(
       {"rewards-to-go": ret_rtg, "GAE": ret_gae},
       title="Task 3: rewards-to-go vs GAE",
    )
    # plot_learning_curves(
        # {"GAE": ret_gae},
        # title="Task 3: rewards-to-go vs GAE",
    # )
    # record_video(policy_gae, path="videos/task3_gae.mp4")  # optional