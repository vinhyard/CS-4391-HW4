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


#----------------------------------------------------------------------------
#   TESTING ENSEMBLE CRITICS
#----------------------------------------------------------------------------

def build_critic2(state_dim, hidden_size):
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

def build_critic3(state_dim, hidden_size):
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
def build_critic4(state_dim, hidden_size):
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
def build_critic5(state_dim, hidden_size):
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
def build_critic6(state_dim, hidden_size):
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

        Implementation:
        delta_t = rewards + gamma * next_values * (1 - done_t) - values

        A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

        Implementation:
        Isn't A approximated as  = E{r + gamma * (next_values) - values}
        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    """
    # print(dones)
    # print(type(dones))
    # print(dones.shape)
    delta_t = rewards + gamma * next_values * (1-dones) - values
    advantages_first_version = delta_t + gamma * lam * (1-dones) * (rewards + gamma*(next_values)-values)
    # print(f"delta_t: {delta_t}")
    # print(f"gamma: {gamma}")
    # print(f"lam: {lam}")
    # print(f"dones: {1-dones}")

    #                                                   r     + gamma (V'(s'))  - V(s)
    # for delta in delta_t:
    #   print(delta)
    # paper shows A_t = delta_t + lam * gamma *delta_t+1 + ... + (lam * gamma)^T-t+1 
    return advantages_first_version


def reinforce_adv_signal(policy, states, actions, advantages):
    """Policy-gradient loss weighted by arbitrary advantages (e.g. GAE)."""
    # TODO: compute  -E[ A_t * log pi(a | s) ].
    loss = advantages * _log_prob(policy=policy,states=states,actions=actions)
    # print(loss.shape)
    # print(loss)
    return loss.mean()


def train_advantage_vpg(
    epochs=3,
    episodes=20,
    updates=10,
    critic_updates=80,
    learning_rate=1e-4,
    critic_lr=3e-4,
    hidden_size=32,
    layers=2,
    batch_size=500,
    gamma=0.975,
    lam=1,
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
    critic_2     = build_critic2(state_dim, hidden_size)
    critic_3     = build_critic3(state_dim,hidden_size)
    critic_4     = build_critic4(state_dim,hidden_size)
    critic_5     = build_critic5(state_dim,hidden_size)
    critic_6     = build_critic6(state_dim,hidden_size)


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
        # buffer.to(device)

        # --- train the critic ---
        # Regress V(s) toward the reward-to-go targets for critic_updates steps.
        for _ in range(critic_updates):
            states, _, _, _, _, rtg, _ = buffer.sample(batch_size)
            states_t = torch.as_tensor(states, dtype=torch.float32)
            rtg_t    = torch.as_tensor(rtg,    dtype=torch.float32)
            cr_optimizer.zero_grad()

            # TODO: compute mse_loss between critic(states_t) and rtg_t,
            #       then call .backward() and cr_optimizer.step().
            mse = mse_loss(critic(states_t),rtg_t).backward()
            # mse2 = mse_loss(critic_2(states_t),rtg_t).backward()
            # mse3 = mse_loss(critic_3(states_t),rtg_t).backward()

            cr_optimizer.step()

        # --- compute GAE advantages ---
        # Run the critic (no gradients) on every stored state.
        all_states = torch.as_tensor(buffer.states[: buffer.max_i], dtype=torch.float32)
        with torch.no_grad():
            values_1 = critic(all_states).numpy()          # V(s_t)
            # values_2 = critic_2(all_states).numpy()          # V2(s_t)
            # values_3 = critic_3(all_states).numpy()          # V3(s_t)
            # values_4 = critic_4(all_states).numpy()          # V3(s_t)
            # values_5 = critic_5(all_states).numpy()          # V3(s_t)
            # values_6 = critic_6(all_states).numpy()          # V3(s_t)

            values = values_1

        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]                    # V(s_{t+1}), 0 at episode end

        # TODO: call compute_gae(...) to get an (N, 1) array of advantages.

        advantages = compute_gae(rewards=buffer.rewards,values = values, next_values = next_values, dones = buffer.dones)  # TODO
        advantages = np.array(advantages)

        # Normalise for training stability (provided).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- train the actor ---
        for _ in range(updates):
            idxs      = np.random.randint(0, buffer.max_i, size=batch_size)
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

# '''
if __name__ == "__main__":
    # from plotting import plot_learning_curves
    # from video import record_video
    # from vpg import train_vpg

    # Example: compare rewards-to-go vs GAE
    # policy_rtg, ret_rtg = train_vpg(epochs=200, learning_rate=3e-4)
    policy_gae, ret_gae1 = train_advantage_vpg(epochs=200, learning_rate=3e-4, lam=1)
    # policy_gae, ret_gae5 = train_advantage_vpg(epochs=100, learning_rate=3e-4, lam=0.5)
    policy_gae, ret_gae0 = train_advantage_vpg(epochs=200, learning_rate=3e-4, lam=0)

    '''
    plot_learning_curves(
       {"rewards-to-go": ret_rtg, "GAE": ret_gae},
       title="Task 3: rewards-to-go vs GAE",
    )
    '''
    # plot_learning_curves(
    #     {"GAE lam 1": ret_gae1, "GAE lam 0": ret_gae0},
    #     title="Task 3: GAE lam1 vs GAE lam.5 vs GAE lam0",
    # )
    # record_video(policy_gae, path="videos/task3_gae.mp4")  # optional
# '''