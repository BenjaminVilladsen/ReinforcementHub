packaging = "sarsa"
import time

import numpy as np
import gymnasium as gym
import signal

from file_handling import store_policy
from utils import print_episode_stats, print_text_with_border


def sarsa(epsilon_greedy_policy_fn, discretize_fn, print_fn, q_table, env, settings, bins, epsilon_decay=True):
    def save_policy_on_interrupt(signum, frame):
        filename = f"interrupt_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        should_save = input(f"\nInterrupt detected. Do you want to save the policy to {filename}? (y/n): ").strip().lower()
        if should_save == 'y':
            store_policy(filename, q_table, settings)
            print(f"\nPolicy saved to {filename} on interrupt.")
        print_text_with_border("Exiting on user interrupt.", px=40, py=0)
        exit(0)

    # Set up the signal handler
    signal.signal(signal.SIGINT, save_policy_on_interrupt)

    success_count = 0
    convergence_count = 0
    start_time = time.time()
    episode_rewards = []
    current_epsilon = settings['epsilon']
    prev_avg = -1_000_000
    curr_avg = -1_000_00
    for episode in range(settings['num_episodes']):
        initial_state = env.reset()
        current_state = discretize_fn(initial_state, bins)
        current_action = epsilon_greedy_policy_fn(current_state, settings['epsilon'], env, q_table)
        done = False
        truncated = False

        episode_reward = 0
        t = 0

        while not done:
            t += 1
            next_state_raw, reward, done, truncated, _ = env.step(current_action)  # Environment step
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


        curr_avg = np.mean(episode_rewards) #update current average
        avg_diff = abs(curr_avg) - abs(prev_avg)

        if abs(avg_diff) < settings['convergence_threshold']:

            convergence_count += 1
        else:
            convergence_count = 0 #reset convergence count if no convergence. We want consequitive convergence
        prev_avg = curr_avg

        # Check success criteria
        if done and episode_reward >= settings['success_threshold']:
            success_count += 1

        if episode % settings['log_interval'] == 0 and episode > 0:
            # print average reward the most recent 100 episodes
            print_fn(episode_rewards, title=episode, settings=settings, convergence_count=convergence_count, success_count=success_count, time_elapsed=time.time() - start_time)
            print("Epsilon: ", current_epsilon)

        #Update current epsilon
        if epsilon_decay:
            current_epsilon = max(settings['epsilon_min'], current_epsilon * settings['epsilon_decay'])

        if convergence_count >= settings['convergence_count_limit']:
            print_text_with_border("Convergence reached!", px=40, py=1)
            print("Exiting training loop.")
            title = f"Convergence at episode: {episode}"
            print_fn(episode_rewards, title=title, settings=settings, convergence_count=convergence_count,
                     success_count=success_count, time_elapsed=time.time() - start_time)
            print("Epsilon: ", current_epsilon)
            break

    env.close()
    return q_table, episode_rewards
