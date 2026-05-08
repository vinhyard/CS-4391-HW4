"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

from buffer import Buffer, collect_data, act, rescale_actions
from vpg import _log_prob, build_actor
from gae import build_critic, compute_gae


"""
HW4 — Tasks 4 & 5: PPO surrogate objective and full PPO algorithm.

Depends on: buffer.py (Task 1), vpg.py (Task 2), gae.py (Task 3).
"""

import numpy as np
import torch as th
from torch.nn.functional import mse_loss
from torch.distributions import Normal
import gymnasium as gym

# from buffer import Buffer, collect_data, act, rescale_actions
# from vpg import _log_prob, build_actor
# from gae import build_critic, compute_gae


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
):
    """Full PPO algorithm with a single actor (N = 1).

    Returns:
        policy  — the trained actor network (pass to video.record_video)
        returns — list of per-iteration average episodic returns
        losses  — list of per-iteration total loss values
    """
    env = gym.make("Pendulum-v1")
    state_dim  = env.reset()[0].shape[0]
    action_dim = env.action_space.sample().shape[0]
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
        for _ in range(steps_per_iter):
            states_c, actions_c, rewards_c, next_states_c, dones_c, rtg, _, _ = buffer.sample(steps_per_iter)
            states_c_t = th.as_tensor(states_c, dtype=th.float32)
            rtg_t    = th.as_tensor(rtg,    dtype=th.float32)
        cr_optimizer.zero_grad()
        mse = mse_loss(critic(s_t),rtg_t).backward()
        cr_optimizer.step()
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
            print(mini_advantages.shape)
            print(mini_states_t.shape)
            print(mini_actions_t.shape)
            print(mini_returns.shape)


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
            optimizer.step()
            prev_loss = mini_ppo_total_loss.item()          
        returns_per_iter.append(np.mean(returns))
        losses_per_iter.append(prev_loss)
        print(f"{k+1}/{iterations} iterations  loss={prev_loss:.4f}")


        #=================
        # Normalise for training stability (provided).
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #===============

    # TODO: return policy, list_of_returns, list_of_losses
    return policy, returns_per_iter, losses_per_iter


if __name__ == "__main__":
    from plotting import plot_learning_curves, plot_loss_curves
    from video import record_video, generate_strobe

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
    # record_video(policy, path="videos/task5_ppo.mp4")            # optional
    # generate_strobe(policy, path="videos/task5_ppo_strobe.png")  # optional