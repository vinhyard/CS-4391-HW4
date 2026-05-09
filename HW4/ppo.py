"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.
Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""
import time
import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym
import torch.nn as nn
from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor, _log_prob_recurrent
from gae import build_critic, compute_gae
from gymnasium.spaces import Box

class AngVel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=-8.0, high=8.0, shape=(1,), dtype=np.float32)
    def observation(self, obs):
        return np.asarray([obs[2]], dtype=np.float32)   # full obs is [cos θ, sin θ, θ̇]

def make_partial_env():
    return AngVel(gym.make("Pendulum-v1"))

# ---------------------------------------------------------------------------
# Internal helper (provided — do not modify)
# ---------------------------------------------------------------------------
def _critic_values(critic, buffer):
    states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
    with th.no_grad():
        values = critic(states).numpy()
    next_values = np.zeros_like(values)
    next_values[:-1] = values[1:]
    return values, next_values

# ---------------------------------------------------------------------------
# Task 4
# ---------------------------------------------------------------------------
def ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs, eps_clip=0.2, clip=True):
    log_pi = _log_prob(policy, states, actions)
    probability_ratio = th.exp(log_pi - old_log_probs)
    surrogate = probability_ratio * advantages
    if clip == True:
        clipped_surrogate = th.clamp(probability_ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        loss = -th.min(surrogate, clipped_surrogate).mean()
    else:
        loss = -surrogate.mean()
    return loss

def ppo_total_loss(policy, critic, states, actions, advantages, returns, old_log_probs,
                   eps_clip=0.2, c1=0.5, c2=0.01, clip=True):
    surrogate = ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs,
                                   eps_clip=eps_clip, clip=clip)
    vals = critic(states)
    val_loss = mse_loss(vals, returns)
    mu, sigma = policy(states)
    s_pi = Normal(mu, sigma).entropy().sum(dim=1).mean()
    return surrogate + c1 * val_loss - c2 * s_pi

def collect_data_parallel(total_size, env, agent, num_envs, title="collecting"):
    buffer = Buffer(sdim=3, adim=1, size=total_size)
    rewards = []
    obs, _ = env.reset()
    for _ in range(total_size // num_envs):
        with th.no_grad():
            mu, sigma = agent(th.as_tensor(obs, dtype=th.float32))
            a = rescale_actions(Normal(mu, sigma).sample(), -2, 2).numpy()
        next_obs, r, term, trunc, _ = env.step(a)
        done = np.logical_or(term, trunc)
        for e in range(num_envs):
            buffer.add(obs[e], a[e], r[e], done[e])
            rewards.append(r[e])
        obs = next_obs
    buffer.calc_reward_to_go()
    return buffer, np.mean(rewards)



class StateDependentMLPPolicy(nn.Module):

    LOG_SIGMA_MIN, LOG_SIGMA_MAX = -2.0, 0.5     # tightened to prevent σ collapse

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.mu_head        = nn.Linear(hidden_size, action_dim)
        self.log_sigma_head = nn.Linear(hidden_size, action_dim)
        # zero-init the log-sigma head so it starts at log_sigma = 0 (sigma = 1)
        # for every state, then learns to vary as training progresses
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, s):
        h = self.trunk(s)
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)
        sigma     = log_sigma.exp()
        return mu, sigma


# ---------------------------------------------------------------------------
# Task 5
# ---------------------------------------------------------------------------
def train_ppo(
    iterations=200,
    steps_per_iter=2048,
    sgd_epochs=10,
    minibatch_size=64,
    learning_rate=3e-4,
    hidden_size=64,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
    num_envs=1,
    state_dep_sigma=False,                              
):
    if num_envs > 1:
        env = gym.make_vec("Pendulum-v1", num_envs=num_envs, vectorization_mode="async")
    else:
        env = gym.make("Pendulum-v1")
    state_dim  = 3
    action_dim = 1
    episode_len = env.spec.max_episode_steps

    if state_dep_sigma:
        policy = StateDependentMLPPolicy(state_dim, action_dim, hidden_size)
    else:
        policy = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = th.optim.Adam(params=policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(params=critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []

    for k in range(iterations):
        with th.no_grad():
            if num_envs > 1:
                buffer, avg_rwd = collect_data_parallel(
                    steps_per_iter, env, policy, num_envs, title=f"ppo {k+1}/{iterations}"
                )
            else:
                buffer, avg_rwd = collect_data(
                    steps_per_iter, env, policy, title=f"ppo {k+1}/{iterations}"
                )

        buffer.calc_reward_to_go(gamma=gamma)
        s_t = th.as_tensor(buffer.states, dtype=th.float32)
        a_t = th.as_tensor(buffer.actions, dtype=th.float32)

        all_states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
        with th.no_grad():
            values = critic(all_states).numpy()
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]

        with th.no_grad():
            advantages = compute_gae(rewards=buffer.rewards, values=values,
                                     next_values=next_values, dones=buffer.dones)
        returns = advantages + values

        old_policy = _log_prob(policy=policy, actions=a_t, states=s_t).detach()

        prev_loss = 0.0
        for x in range(sgd_epochs):
            for _ in range(minibatch_size):
                mini_states, mini_actions, mini_rewards, mini_states, mini_dones, mini_rtg, _, mini_idx = buffer.sample(minibatch_size)
                mini_states_t = th.as_tensor(mini_states, dtype=th.float32)
                mini_actions_t = th.as_tensor(mini_actions, dtype=th.float32)
                mini_rtg_t = th.as_tensor(mini_rtg, dtype=th.float32)
                cr_optimizer.zero_grad()
                mse_loss(critic(mini_states_t), mini_rtg_t).backward()
                th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                cr_optimizer.step()

            mini_all_states = th.as_tensor(mini_states_t, dtype=th.float32)
            with th.no_grad():
                mini_values = critic(mini_all_states).detach().numpy()
            mini_next_values = np.zeros_like(mini_values)
            mini_next_values[:-1] = mini_values[1:]

            mini_advantages = compute_gae(rewards=mini_rewards, values=mini_values,
                                          next_values=mini_next_values, dones=mini_dones)
            mini_returns = mini_advantages + mini_values

            mini_adv_t = th.as_tensor(mini_advantages, dtype=th.float32).squeeze(-1)
            mini_ret_t = th.as_tensor(mini_returns,    dtype=th.float32)
            mini_old   = old_policy[mini_idx]

            optimizer.zero_grad()
            mini_ppo_total_loss = ppo_total_loss(
                policy, critic,
                mini_states_t, mini_actions_t,
                mini_adv_t, mini_ret_t, mini_old,
                eps_clip=eps_clip, c1=c1, c2=c2, clip=clip,
            )
            mini_ppo_total_loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            prev_loss = mini_ppo_total_loss.item()
            optimizer.step()

        returns_per_iter.append(np.mean(returns))
        losses_per_iter.append(prev_loss)
        print(f"{k+1}/{iterations} iterations  loss={prev_loss:.4f}")

    return policy, returns_per_iter, losses_per_iter



class RecurrentPolicy(nn.Module):

    LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0

    def __init__(self, obs_dim=1, action_dim=1, hidden_size=64, state_dep_sigma=True):
        super().__init__()
        self.gru     = nn.GRUCell(obs_dim, hidden_size)
        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.state_dep_sigma = state_dep_sigma
        if state_dep_sigma:
            self.log_sigma_head = nn.Linear(hidden_size, action_dim)
        else:
            self.log_sigma = nn.Parameter(th.zeros(action_dim))
        self.hidden_size = hidden_size

    def forward(self, obs, hidden):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h  = self.gru(obs, hidden)
        mu = self.mu_head(h)
        if self.state_dep_sigma:
            log_sigma = self.log_sigma_head(h).clamp(self.LOG_SIGMA_MIN, self.LOG_SIGMA_MAX)
            sigma = log_sigma.exp()
        else:
            sigma = self.log_sigma.exp().expand_as(mu)
            sigma = th.clamp(sigma, min=1e-3, max=2.0)
        return mu, sigma, h

    def initial_hidden(self, batch=1):
        return th.zeros(batch, self.hidden_size)


def collect_data_recurrent(size, env, policy, title="collecting"):
    obs_dim = env.observation_space.shape[0]
    buffer  = Buffer(sdim=obs_dim, adim=1, size=size)
    hiddens = np.zeros((size, policy.hidden_size), dtype=np.float32)
    rewards = []
    obs, _ = env.reset()
    h = policy.initial_hidden(batch=1)
    for i in range(size):
        with th.no_grad():
            x = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            mu, sigma, new_h = policy(x, h)
            a_t = Normal(mu, sigma).sample()
            a   = rescale_actions(a_t, -2, 2).numpy().squeeze(0)
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.add(obs, a, r, done)
        hiddens[i] = h.squeeze(0).numpy()
        rewards.append(r)
        if done:
            obs, _ = env.reset()
            h = policy.initial_hidden(batch=1)
        else:
            obs = next_obs
            h   = new_h
    buffer.calc_reward_to_go()
    return buffer, np.mean(rewards), hiddens


def ppo_surrogate_loss_recurrent(policy, states, actions, advantages, old_log_probs, hidden,
                                 eps_clip=0.2, clip=True):
    log_pi = _log_prob_recurrent(policy, states, actions, hidden)
    ratio  = th.exp(log_pi - old_log_probs)
    surr   = ratio * advantages
    if clip:
        clipped = th.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        return -th.min(surr, clipped).mean()
    return -surr.mean()


def ppo_total_loss_recurrent(policy, critic, states, actions, advantages, returns,
                             old_log_probs, hidden, eps_clip=0.2, c1=0.5, c2=0.01, clip=True):
    surr = ppo_surrogate_loss_recurrent(
        policy, states, actions, advantages, old_log_probs, hidden,
        eps_clip=eps_clip, clip=clip,
    )
    vals = critic(states)
    val_loss = mse_loss(vals, returns)
    mu, sigma, _ = policy(states, hidden)
    ent = Normal(mu, sigma).entropy().sum(dim=-1).mean()
    return surr + c1 * val_loss - c2 * ent


def train_ppo_recurrent(
    iterations=200, steps_per_iter=2048, sgd_epochs=10, minibatch_size=64,
    learning_rate=3e-4, hidden_size=64, gamma=0.99, lam=0.95,
    eps_clip=0.2, c1=0.5, c2=0.01, clip=True,
    state_dep_sigma=True,                                # <-- new
):
    env = make_partial_env()
    obs_dim, action_dim = 1, 1
    policy = RecurrentPolicy(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_size=hidden_size, state_dep_sigma=state_dep_sigma,
    )
    critic       = build_critic(obs_dim, hidden_size)
    optimizer    = th.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)
    returns_per_iter, losses_per_iter = [], []
    for k in range(iterations):
        with th.no_grad():
            buffer, avg_rwd, hiddens_np = collect_data_recurrent(
                steps_per_iter, env, policy, title=f"ppo-rnn {k+1}/{iterations}"
            )
        buffer.calc_reward_to_go(gamma=gamma)
        s_t = th.as_tensor(buffer.states,  dtype=th.float32)
        a_t = th.as_tensor(buffer.actions, dtype=th.float32)
        h_t = th.as_tensor(hiddens_np,     dtype=th.float32)

        with th.no_grad():
            values = critic(s_t).numpy()
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:]
        advantages = compute_gae(
            rewards=buffer.rewards, values=values,
            next_values=next_values, dones=buffer.dones,
            gamma=gamma, lam=lam,
        )
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        adv_t_full = th.as_tensor(advantages, dtype=th.float32).squeeze(-1)
        ret_t_full = th.as_tensor(returns,    dtype=th.float32)
        with th.no_grad():
            old_log_probs = _log_prob_recurrent(policy, s_t, a_t, h_t).detach()

        prev_loss = 0.0
        N = buffer.max_i
        for _ in range(sgd_epochs):
            for _ in range(minibatch_size):
                idx = np.random.randint(0, N, size=minibatch_size)
                mini_s   = s_t[idx]
                mini_a   = a_t[idx]
                mini_h   = h_t[idx]
                mini_adv = adv_t_full[idx]
                mini_ret = ret_t_full[idx]
                mini_old = old_log_probs[idx]
                optimizer.zero_grad()
                cr_optimizer.zero_grad()
                loss = ppo_total_loss_recurrent(
                    policy, critic,
                    mini_s, mini_a, mini_adv, mini_ret, mini_old, mini_h,
                    eps_clip=eps_clip, c1=c1, c2=c2, clip=clip,
                )
                loss.backward()
                th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                optimizer.step()
                cr_optimizer.step()
                prev_loss = loss.item()
        ep_return = avg_rwd * 200
        returns_per_iter.append(ep_return)
        losses_per_iter.append(prev_loss)
        print(f"ppo-rnn {k+1}/{iterations}: return={ep_return:.2f}  loss={prev_loss:.4f}")
    return policy, returns_per_iter, losses_per_iter



if __name__ == "__main__":
    from plotting import plot_learning_curves, plot_loss_curves
    from video import record_video, generate_strobe
    for n in [1, 4, 8]:
        t = time.time()
        train_ppo(iterations=50, steps_per_iter=2048, num_envs=n)
        print(f"num_envs={n}: {time.time() - t:.1f}s")
    # --- Task 4: clipped vs unclipped ---
    _, ret_clip,   loss_clip   = train_ppo(iterations=50, clip=True)
    _, ret_noclip, loss_noclip = train_ppo(iterations=50, clip=False)
    plot_learning_curves(
        {"clipped": ret_clip, "unclipped": ret_noclip},
        title="Task 4: PPO clipped vs unclipped",
    )
    plot_loss_curves(
        {"clipped": loss_clip, "unclipped": loss_noclip},
        title="Task 4: PPO loss curves",
    )
    _, ret_indep, _ = train_ppo(iterations=50, state_dep_sigma=False)
    _, ret_dep,   _ = train_ppo(iterations=50, state_dep_sigma=True)
    plot_learning_curves(
        {"fixed log-sigma":         ret_indep,
         "state-dependent log-sigma": ret_dep},
        title="Extension: state-dependent log-σ(s)",
        smooth=0.9,
    )
    # --- Task 5: full PPO ---
    policy, ret_ppo, loss_ppo = train_ppo(iterations=500)
    plot_learning_curves({"PPO": ret_ppo}, title="Task 5: Full PPO")
    plot_loss_curves({"PPO": loss_ppo}, title="Task 5: Total loss")
    record_video(policy, path="videos/task5_ppo.mp4")            # optional
    generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional
    _, ret_full,   _ = train_ppo(iterations=50)
    _, ret_rnn_pomdp, _ = train_ppo_recurrent(iterations=50)

    plot_learning_curves(
        {"MLP + full state": ret_full,
         "GRU + ang-vel only": ret_rnn_pomdp},
        title="Extension: partial obs with recurrence",
        smooth=0.9,
    )

