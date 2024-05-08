import numpy as np
import gym

from config_and_helpers import initializeQAndEnv, epsilon_greedy_policy, discretize, n_bins, state_bounds


def update_q(Q, state, action, reward, next_state, next_action, alpha, gamma, done):
  current_q = Q[state][action]
  future_q = Q[next_state][next_action] if not done else 0
  updated_q = current_q + alpha * (reward + gamma * future_q - current_q)
  Q[state][action] = updated_q
  return Q


def sarsa(num_episodes, alpha, gamma, epsilon):
    """
    SARSA algorithm.

    Parameters:
        num_episodes (int): Number of episodes to run.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Epsilon value for exploration-exploitation trade-off.

    Returns:
        np.array: Learned Q-table.
    """
    Q, env, nA = initializeQAndEnv()
    for i_episode in range(num_episodes):
        state = env.reset()
        state = discretize(state, n_bins, bounds=state_bounds)
        action = epsilon_greedy_policy(state, Q, epsilon, env)
        total_reward = 0

        while True:
            next_state, reward, done, _ = env.step(action)
            next_state = discretize(next_state, n_bins, bounds=state_bounds)
            next_action = epsilon_greedy_policy(next_state, Q, epsilon, env)

            # SARSA update
            Q[state + (action,)] += alpha * (reward + gamma * Q[next_state + (next_action,)] - Q[state + (action,)])

            total_reward += reward
            state, action = next_state, next_action

            if done:
                break

        print(f"Episode {i_episode + 1}: Total Reward = {total_reward}")

    return Q