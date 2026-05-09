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
    """Run the critic on every stored state, returning (values, next_values)."""
    states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
    with th.no_grad():
        values = critic(states).numpy()
    next_values = np.zeros_like(values)
    next_values[:-1] = values[1:]
    return values, next_values


# ---------------------------------------------------------------------------
# Task 4 TODOs
# ---------------------------------------------------------------------------

def ppo_surrogate_loss(
    policy, states, actions, advantages, old_log_probs, eps_clip=0.2, clip=True
):
    """PPO surrogate objective (PPO paper, Equation 7).

        r_t(theta) = exp( log pi_theta(a|s) - log pi_theta_old(a|s) )

        unclipped:   L = E[ r_t * A_t ]
        clipped:     L = E[ min( r_t * A_t,
                                 clip(r_t, 1 - eps, 1 + eps) * A_t ) ]

    Returns the *negative* of the objective so that optimizer.step()
    performs gradient ascent on the expected return.
    """
    log_pi = _log_prob(policy, states, actions)
    probability_ratio = th.exp(log_pi - old_log_probs)
    surrogate = probability_ratio * advantages
    if clip == True:
        clipped_surrogate = th.clamp(probability_ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        loss = -th.min(surrogate, clipped_surrogate).mean()
    else:
        loss = -surrogate.mean()
    return loss



def ppo_total_loss(
    policy,
    critic,
    states,
    actions,
    advantages,
    returns,
    old_log_probs,
    eps_clip=0.2,
    c1=0.5,
    c2=0.01,
    clip=True,
):
    """PPO total loss (PPO paper, Equation 9).

        L_total = L_surr  +  c1 * L_VF  -  c2 * S[pi]

    where L_VF = ( V_theta(s) - R_t )^2  and  S[pi] is the policy entropy.
    Returns a scalar tensor to be minimised.
    """

    surrogate = ppo_surrogate_loss(policy, states, actions, advantages, old_log_probs,
                                   eps_clip=eps_clip, clip=clip)
    vals = critic(states)
    val_loss = mse_loss(vals, returns)
    mu, sigma = policy(states)
    s_pi = Normal(mu, sigma).entropy().sum(dim = 1).mean()
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
# ---------------------------------------------------------------------------
# Task 5 TODO
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
):
    """Full PPO algorithm with a single actor (N = 1).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-iteration average episodic returns
        losses  — list of per-iteration total loss values
    """
    if num_envs > 1:
        env = gym.make_vec("Pendulum-v1", num_envs = num_envs, vectorization_mode="async")
    else:
        env = gym.make("Pendulum-v1")
    state_dim  = 3
    action_dim = 1
    episode_len = env.spec.max_episode_steps

    policy       = build_actor(state_dim, action_dim, hidden_size)
    critic       = build_critic(state_dim, hidden_size)
    optimizer    = th.optim.Adam(policy.parameters(), lr=learning_rate)
    cr_optimizer = th.optim.Adam(critic.parameters(), lr=learning_rate)

    returns_per_iter = []
    losses_per_iter  = []
    
    for k in range(iterations):
        # TODO: 1) roll out the current policy for `steps_per_iter` steps
        #          and store transitions in a Buffer.
        #       2) compute V(s) and V(s') with the critic, then GAE advantages
        #          and target returns (returns = advantages + V(s)).
        #       3) cache the log-probabilities of the sampled actions under
        #          the *old* policy (detach from the graph).
        #       4) for `sgd_epochs` epochs, iterate over minibatches of the
        #          collected data and minimise ppo_total_loss(...).
        #       5) log per-iteration episodic return and total loss for the
        #          required learning / loss curve plots.



        #====================================================================
        # TODO: 1) roll out the current policy for `steps_per_iter` steps
        #          and store transitions in a Buffer.
        #====================================================================

        with th.no_grad():
            if num_envs > 1:
                buffer, avg_rwd = collect_data_parallel(
                    steps_per_iter, env, policy, num_envs, title=f"ppo {k+1}/{iterations}"
                )
            else:
                buffer, avg_rwd = collect_data(
                steps_per_iter, env, policy, title=f"gae {k + 1}/{iterations}"
            )

        # TODO: fill buffer.ret_to_go using buffer.calc_reward_to_go(gamma).
        buffer.calc_reward_to_go(gamma=gamma)
        reward_to_go_t = th.as_tensor(buffer.ret_to_go, dtype=th.float32)
        s_t = th.as_tensor(buffer.states, dtype=th.float32)
        a_t = th.as_tensor(buffer.actions, dtype=th.float32)



        #====================================================================
        #       2) compute V(s) and V(s') with the critic, then GAE advantages
        #          and target returns (returns = advantages + V(s)).
        #====================================================================

        # --- train the critic ---
        #Regress V(s) toward the reward-to-go targets for critic_updates steps.

        # --- compute GAE advantages ---
        # Run the critic (no gradients) on every stored state.
        all_states = th.as_tensor(buffer.states[: buffer.max_i], dtype=th.float32)
        with th.no_grad():
            values = critic(all_states).numpy()          # V(s_t)
        next_values = np.zeros_like(values)
        next_values[:-1] = values[1:] 
                  # V(s_{t+1}), 0 at episode end

        # compute_gae(...) to get an (N, 1) array of advantages.
        with th.no_grad():
          advantages = compute_gae(rewards=buffer.rewards,values = values, next_values = next_values, dones = buffer.dones)  # TODO
        returns = advantages + values


        #====================================================================
        #       3) cache the log-probabilities of the sampled actions under
        #          the *old* policy (detach from the graph).
        #====================================================================

        old_policy = _log_prob(policy=policy,actions=a_t,states=s_t).detach()


        #====================================================================
        #       4) for `sgd_epochs` epochs, iterate over minibatches of the
        #          collected data and minimise ppo_total_loss(...).
        #====================================================================

        prev_loss = 0.0
        for x in range(sgd_epochs):

            # --- train the critic ---
            # Regress V(s) toward the reward-to-go targets for critic_updates steps.

            for _ in range(minibatch_size):
                mini_states, mini_actions, mini_rewards, mini_states, mini_dones, mini_rtg, _, mini_idx = buffer.sample(minibatch_size)
                mini_states_t = th.as_tensor(mini_states, dtype=th.float32)
                mini_actions_t = th.as_tensor(mini_actions, dtype=th.float32)
                mini_rtg_t    = th.as_tensor(mini_rtg,    dtype=th.float32)
                cr_optimizer.zero_grad()
                mse_mini = mse_loss(critic(mini_states_t),mini_rtg_t).backward()

                cr_optimizer.step()

            # --- compute GAE advantages ---
            # Run the critic (no gradients) on every stored state.
            mini_all_states = th.as_tensor(mini_states_t, dtype=th.float32)
            with th.no_grad():
                mini_values = critic(mini_all_states).detach().numpy()          # V(s_t)
            mini_next_values = np.zeros_like(mini_values)
            mini_next_values[:-1] = mini_values[1:]                    # V(s_{t+1}), 0 at episode end

            # compute_gae(...) to get an (N, 1) array of advantages.

            mini_advantages = compute_gae(rewards=mini_rewards,values = mini_values, next_values = mini_next_values, dones = mini_dones)  # TODO
            mini_returns = mini_advantages + mini_values



            mini_adv_t = th.as_tensor(mini_advantages, dtype=th.float32).squeeze(-1)
            mini_ret_t = th.as_tensor(mini_returns,    dtype=th.float32)
            mini_old   = old_policy[mini_idx]

            optimizer.zero_grad()
            mini_ppo_total_loss = ppo_total_loss(
                policy, critic,
                mini_states_t, mini_actions_t,
                mini_adv_t, mini_ret_t, mini_old,
                eps_clip=eps_clip, c1=c1, c2=c2, clip=clip,)
            mini_ppo_total_loss.backward()
            prev_loss = mini_ppo_total_loss.item()          
            optimizer.step()
        returns_per_iter.append(np.mean(returns))
        losses_per_iter.append(prev_loss)
        print(f"{k+1}/{iterations} iterations  loss={prev_loss:.4f}")


        #=================
        # Normalise for training stability (provided).
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #===============

    # TODO: return policy, list_of_returns, list_of_losses
    return policy, returns_per_iter, losses_per_iter

class RecurrentPolicy(nn.Module):

    def __init__(self, obs_dim=1, action_dim=1, hidden_size=64):
        super().__init__()
        self.gru       = nn.GRUCell(obs_dim, hidden_size)
        self.mu_head   = nn.Linear(hidden_size, action_dim)
        self.log_sigma = nn.Parameter(th.zeros(action_dim))   # state-independent std
        self.hidden_size = hidden_size

    def forward(self, obs, hidden):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.gru(obs, hidden)
        mu = self.mu_head(h)
        sigma = self.log_sigma.exp().expand_as(mu)
        sigma = th.clamp(sigma, min=1e-3, max=2.0)
        return mu, sigma, h

    def initial_hidden(self, batch=1):
        return th.zeros(batch, self.hidden_size)


def collect_data_recurrent(size, env, policy, title="collecting"):
    """Single-env rollout that carries hidden state across timesteps.

    Returns (buffer, avg_reward, hiddens_np). `hiddens_np[t]` is the hidden
    state that was *fed in* at step t (i.e. h_t, before the GRUCell update).
    """
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


def ppo_surrogate_loss_recurrent(
    policy, states, actions, advantages, old_log_probs, hidden,
    eps_clip=0.2, clip=True,
):
    log_pi = _log_prob_recurrent(policy, states, actions, hidden)
    ratio  = th.exp(log_pi - old_log_probs)
    surr   = ratio * advantages
    if clip:
        clipped = th.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages
        return -th.min(surr, clipped).mean()
    return -surr.mean()


def ppo_total_loss_recurrent(
    policy, critic, states, actions, advantages, returns, old_log_probs, hidden,
    eps_clip=0.2, c1=0.5, c2=0.01, clip=True,
):
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
):

    env = make_partial_env()
    obs_dim, action_dim = 1, 1

    policy = RecurrentPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size)

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
                optimizer.step()
                cr_optimizer.step()
                prev_loss = loss.item()

        ep_return = avg_rwd * 200                          # Pendulum episode length
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

