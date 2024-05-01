import numpy as np
import time
from config_and_helpers import (
    env, discretize, epsilon_greedy_policy, update_Q_n_step, n_bins, state_bounds, alpha_values, gamma_values,
    epsilon_decay_values, n_episodes, nA
)
from file_operations import store_policy, store_hyperparameters

def train(n=3):
    best_score = -float('inf')
    best_hyperparameters = None
    best_Q = None

    for alpha in alpha_values:
        for gamma in gamma_values:
            for epsilon_decay in epsilon_decay_values:
                print("\n")
                print(f"Running for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}")
                epsilon = 1.0
                epsilon_min = 0.01
                Q = np.zeros((n_bins,) * 8 + (nA,))  # Reset Q-table for each set of hyperparameters

                episode_rewards = []
                episode_lengths = []
                rewards_above_zero = 0
                for i_episode in range(n_episodes):
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    observation, info = env.reset()
                    state = discretize(observation, n_bins, state_bounds)
                    action = epsilon_greedy_policy(state, Q, epsilon)
                    total_reward = 0
                    steps = 0


                    states = [state]
                    actions = [action]
                    rewards = []

                    while True:
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
                            Q = update_Q_n_step(Q, states[0], actions[0], G, states[n], actions[n], alpha, gamma, (done or truncated))
                            # Remove the oldest transition
                            states.pop(0)
                            actions.pop(0)
                            rewards.pop(0)

                        if done or truncated:
                            break
                        state, action = next_state, next_action

                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)

                    #if rewards are above 0
                    if (total_reward > 0):
                        rewards_above_zero += 1

                    if total_reward > best_score:
                        best_score = total_reward
                        best_hyperparameters = {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon_decay}
                        best_Q = Q.copy()

                    if i_episode % 100 == 0:
                        print(
                            f"Episode: {i_episode + 1}, Average Reward: {np.mean(episode_rewards[-(n_episodes // 10):]):.2f}, Average Length: {np.mean(episode_lengths[-(n_episodes // 10):]):.2f}, Total Reward: {sum(episode_rewards[-(n_episodes // 10):]):.2f}, Best Score: {best_score}, rewards > 0: {rewards_above_zero}")
                print(f"Finished training for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay} with best score: {best_score}")

    filename = f"best_policy_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    store_policy(filename, best_Q)
    store_hyperparameters(filename.replace(".pkl", ".json"), best_hyperparameters)
    print(f"Best policy and hyperparameters saved to {filename}")
    return best_Q

if __name__ == "__main__":
    train(n=3)
