import numpy as np
import gym
from config_lunar_lander import num_episodes, num_bins, epsilon, gamma, alpha


def sarsa(epsilon_greedy_policy_fn, discretize_fn, q_table, env, state_bounds):
    bins = [np.linspace(b[0], b[1], num_bins) for b in state_bounds]

    for episode in range(num_episodes):
        initial_state = env.reset()
        current_state = discretize_fn(initial_state, bins)
        current_action = epsilon_greedy_policy_fn(current_state, epsilon, env, q_table)
        done = False

        episode_reward = 0
        while not done:
            next_state_raw, reward, done, _, _ = env.step(current_action)  # Environment step
            episode_reward += reward
            next_state = discretize_fn(next_state_raw, bins)  # Discretize the resulting state
            next_action = epsilon_greedy_policy_fn(next_state, epsilon, env, q_table)  # Choose next action using epsilon-greedy

            # SARSA update
            td_target = reward + gamma * q_table[next_state + (next_action,)]
            td_delta = td_target - q_table[current_state + (current_action,)]
            q_table[current_state + (current_action,)] += alpha * td_delta

            # Move to the next state and action
            current_state, current_action = next_state, next_action

        if episode % 100 == 0:
            print(episode_reward)

    env.close()
    return q_table

print("DONE")
