import time

import numpy as np
import gymnasium as gym

from utils import print_episode_stats


def sarsa(epsilon_greedy_policy_fn, discretize_fn, print_fn, q_table, env, settings, bins):

    success_count = 0
    convergence_count = 0
    start_time = time.time()
    episode_rewards = []
    current_epsilon = settings['epsilon']
    for episode in range(settings['num_episodes']):
        initial_state = env.reset()
        current_state = discretize_fn(initial_state, bins)
        current_action = epsilon_greedy_policy_fn(current_state, settings['epsilon'], env, q_table)
        done = False

        episode_reward = 0
        t = 0

        while not done:
            t += 1
            next_state_raw, reward, done, _, _ = env.step(current_action)  # Environment step
            episode_reward += reward
            next_state = discretize_fn(next_state_raw, bins)  # Discretize the resulting state
            next_action = epsilon_greedy_policy_fn(next_state,  current_epsilon, env,
                                                   q_table)  # Choose next action using epsilon-greedy

            # SARSA update
            td_target = reward + settings['gamma'] * q_table[next_state + (next_action,)]
            td_delta = td_target - q_table[current_state + (current_action,)]
            q_table[current_state + (current_action,)] += settings['alpha'] * td_delta

            # Move to the next state and action
            current_state, current_action = next_state, next_action

        episode_rewards.append(episode_reward)

        # Check success criteria
        if done and episode_reward >= settings['success_threshold']:
            success_count += 1

        # Check convergence criteria
        if np.abs(td_delta) < settings['convergence_threshold']:
            convergence_count += 1

        if episode % settings['log_interval'] == 0 and episode > 0:
            # print average reward the most recent 100 episodes
            print_episode_stats(episode_rewards[-settings['log_interval']:], i_episode=episode, episode_span=settings['log_interval'])
            print("Epsilon: ", current_epsilon)

        #Update current epsilon
        current_epsilon = max(settings['epsilon_min'], current_epsilon * settings['epsilon_decay'])

    env.close()
    return q_table, episode_rewards
