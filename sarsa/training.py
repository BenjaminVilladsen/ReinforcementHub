import numpy as np
import time
from config_and_helpers import (
    env, discretize, epsilon_greedy_policy, update_Q_n_step, n_bins, state_bounds, nA, print_stats
)
from file_operations import store_policy, store_hyperparameters

def train(
    n=3, alpha=0.1, gamma=0.95, n_episodes=800, log_interval=200, max_time_steps=2000,
    init_epsilon=1.0, min_epsilon=0.2, epsilon_decay=0.9999, verbose=True, storeFile=True
):
    # Total best and worst scores across all episodes
    t_best_score = -np.inf
    t_worst_score = np.inf
    episode_rewards = []
    t_rewards_100_count = 0
    t_rewards_200_count = 0

    epsilon = init_epsilon
    if verbose:
        print(f"Running for alpha={alpha}, gamma={gamma}, epsilon={epsilon} decay:{epsilon_decay}, n_episodes={n_episodes}")
    if not storeFile:
        print("WARNING: Not storing Q")

    Q = np.zeros((n_bins,) * 8 + (nA,))  # Initialize Q_table

    for i_episode in range(n_episodes):
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        action = epsilon_greedy_policy(state, Q, epsilon)
        total_reward = 0
        states, actions, rewards = [state], [action], []

        for _ in range(max_time_steps):
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            next_state = discretize(observation, n_bins, state_bounds)
            next_action = epsilon_greedy_policy(next_state, Q, epsilon)

            states.append(next_state)
            actions.append(next_action)

            if len(rewards) >= n:
                G = sum(gamma ** i * rewards[i] for i in range(n))
                Q = update_Q_n_step(Q, states[0], actions[0], G, states[n], actions[n], alpha, gamma, done or truncated)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            if done or truncated:
                break
            state, action = next_state, next_action

        episode_rewards.append(total_reward)
        t_best_score = max(t_best_score, total_reward)
        t_worst_score = min(t_worst_score, total_reward)

        if total_reward >= 100:
            t_rewards_100_count += 1
        if total_reward >= 200:
            t_rewards_200_count += 1

        # Log interval management
        if i_episode % log_interval == 0 and verbose:
            interval_rewards = episode_rewards[-log_interval:]
            e_best_score = max(interval_rewards, default=-np.inf) if interval_rewards else total_reward
            e_worst_score = min(interval_rewards, default=np.inf) if interval_rewards else total_reward
            e_avg_reward = np.mean(interval_rewards) if interval_rewards else 0
            print_stats(
                episode_num=i_episode,
                e_avg_reward=e_avg_reward,
                t_avg_reward=np.mean(episode_rewards),
                e_best=e_best_score,
                t_best=t_best_score,
                e_worst=e_worst_score,
                t_worst=t_worst_score,
                e_above_100_count=sum(reward >= 100 for reward in interval_rewards),
                t_above_100_count=t_rewards_100_count,
                e_above_200_count=sum(reward >= 200 for reward in interval_rewards),
                t_above_200_count=t_rewards_200_count
            )

    if storeFile:
        filename = f"best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q)
        store_hyperparameters(filename.replace(".pkl", ".json"), {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon})

    if verbose:
        print(f"Finished training with total best score: {t_best_score} and total worst score: {t_worst_score}")

    return Q, np.mean(episode_rewards)

if __name__ == "__main__":
    train()
