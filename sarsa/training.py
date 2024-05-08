import numpy as np
import time
from config_and_helpers import (
    discretize, epsilon_greedy_policy, n_bins, state_bounds,
    initializeQAndEnv, plot_rewards
)
from file_operations import store_policy, store_hyperparameters


def train(
        n=3, alpha=0.1, gamma=0.95, n_episodes=800, log_interval=200, max_time_steps=2000,
        init_epsilon=1.0, min_epsilon=0.2, epsilon_decay=0.9999, verbose=True, storeFile=True
):
    episode_rewards = []
    epsilon = init_epsilon
    if verbose:
        print(
            f"Running for alpha={alpha}, gamma={gamma}, epsilon={epsilon} decay:{epsilon_decay}, n={n}, n_bins={n_bins} n_episodes={n_episodes}")
    if not storeFile:
        print("WARNING: Not storing Q")

    Q, env, nA = initializeQAndEnv()

    for i_episode in range(n_episodes):
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        action = epsilon_greedy_policy(state, Q, epsilon, env)
        states, actions, rewards = [state], [action], []

        total_reward = 0
        t = 0
        T = np.inf

        if verbose:
            print(f"Episode #{i_episode + 1}")

        while True:
            if t < T:
                next_observation, reward, done, truncated, info = env.step(action)
                total_reward += reward
                rewards.append(reward)
                next_state = discretize(next_observation, n_bins, state_bounds)
                if done:
                    T = t + 1
                    states.append(None)
                    actions.append(None)
                else:
                    next_action = epsilon_greedy_policy(next_state, Q, epsilon, env)
                    states.append(next_state)
                    actions.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = sum(gamma ** i * rewards[tau + i] for i in range(min(n, T - tau)))
                if tau + n < T:
                    G += gamma ** n * Q[states[tau + n], actions[tau + n]]

                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            if tau == T - 1:
                break

            state, action = next_state, next_action
            t += 1

        epsilon = max(epsilon * epsilon_decay, min_epsilon)
        episode_rewards.append(total_reward)

        if verbose and i_episode % log_interval == 0 and i_episode > 0:
            recent_average_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {i_episode}, Average Reward: {recent_average_reward}")

    if storeFile:
        filename = f"best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q)
        store_hyperparameters(filename.replace(".pkl", ".json"),
                              {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon, 'final_epsilon': epsilon})

    if verbose:
        print(f"Finished training")

    # plot_rewards(episode_rewards)  # This will plot the rewards

    return Q, np.mean(episode_rewards)


if __name__ == "__main__":
    train()
