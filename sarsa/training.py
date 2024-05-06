import numpy as np
import time
from config_and_helpers import (
    env, discretize, epsilon_greedy_policy, update_Q_n_step, n_bins, state_bounds,
    nA, n, print_stats
)
from file_operations import store_policy, store_hyperparameters


############################################################
#                    TRAINING THE MODEL                    #
############################################################

def train(n=3, alpha=0.1, gamma=0.95, n_episodes=800, log_interval=200, max_time_steps=2000, init_epsilon=1.0,
          min_epsilon=0.2, epsilon_decay=0.9999, verbose=True, storeFile=True):
    t_best_score = -float('inf')
    t_worst_score = float('inf')
    best_score = -float('inf')
    worst_score = float('inf')
    best_hyperparameters = None
    best_Q = None
    epsilon = init_epsilon
    if verbose:
        print(f"Running for alpha={alpha}, gamma={gamma}, epsilon={epsilon}, n_episodes={n_episodes}")
    Q = np.zeros((n_bins,) * 8 + (nA,))  # initialize Q_table

    if not storeFile:
        print("WARNING: Not storing Q")

    episode_rewards = []
    episode_lengths = []
    rewards_200_count = 0
    rewards_100_count = 0
    t_rewards_200_count = 0
    t_rewards_100_count = 0
    avg_reward_total = 0
    for i_episode in range(n_episodes):
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        action = epsilon_greedy_policy(state, Q, epsilon)
        total_reward = 0
        steps = 0
        episode_above_zero = 0
        episode_steps = 0

        states = [state]
        actions = [action]
        rewards = []

        for i in range(max_time_steps):
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            next_state = discretize(observation, n_bins, state_bounds)
            next_action = epsilon_greedy_policy(next_state, Q, epsilon)

            states.append(next_state)
            actions.append(next_action)

            if len(rewards) >= n:
                # Calculate discounted return
                G = sum(gamma ** i * rewards[i] for i in range(n))
                # Update Q-table
                Q = update_Q_n_step(Q, states[0], actions[0], G, states[n], actions[n], alpha, gamma,
                                    (done or truncated))
                # Remove the oldest transition
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            if done or truncated:
                break
            state, action = next_state, next_action

            # if rewards are above 0
            if total_reward > 0:
                episode_above_zero += 1
            episode_steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward >= 200:
            t_rewards_200_count += 1
            rewards_100_count += 1
        elif total_reward >= 100:
            rewards_100_count += 1
            t_rewards_100_count += 1

        if total_reward > best_score:
            best_score = total_reward
            best_hyperparameters = {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon}
            best_Q = Q.copy()

        if total_reward > t_best_score:
            t_best_score = total_reward

        if total_reward < worst_score:
            worst_score = total_reward
        if total_reward < t_worst_score:
            t_worst_score = total_reward

        if i_episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-(n_episodes // 10):])
            avg_reward_total = avg_reward_total = np.mean(episode_rewards)
            if (verbose):
                print_stats(
                    episode_num=i_episode,
                    e_avg_reward=avg_reward,
                    t_avg_reward=avg_reward_total,
                    e_best=best_score,
                    t_best=t_best_score,
                    e_worst=worst_score,
                    t_worst=t_worst_score,
                    e_above_100_count=rewards_100_count,
                    t_above_100_count=t_rewards_100_count,
                    e_above_200_count=rewards_200_count,
                    t_above_200_count=t_rewards_200_count
                )
            # reset until next log
            rewards_200_count = 0
            rewards_100_count = 0
            best_score = 0
            worst_score = 0
    if verbose:
        print(
            f"Finished training for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon} with best score: {best_score}")

    if storeFile:
        filename = f"best_policy_{avg_reward_total}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, best_Q)
        store_hyperparameters(filename.replace(".pkl", ".json"), best_hyperparameters)

        if verbose:
            print(f"Best policy and hyperparameters saved to {filename}")
            print(f"avg reward total: {avg_reward_total}")

    return best_Q, avg_reward_total


if __name__ == "__main__":
    train()
