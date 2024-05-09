import json
import pickle
import time
import gymnasium as gym
import numpy as np
from collections import deque
from playsound import playsound

# Hyperparameters
alpha_values = [0.5]# 0.1, 0.05]
gamma_values = [0.5]#, 0.9, 0.99]
epsilon_decay_values = [0.9]#, 1, 0.9]

# Simulation parameters
n_episodes = 1_000
n_bins = 10
n = 20

# Environment setup
env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
nA = env.action_space.n
Q = np.zeros((n_bins,) * 8 + (nA,))

# State bounds
state_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5), (0, 1), (0, 1)]

# Helper functions
def discretize(observation, bins, bounds):
    discretized = [0 if obs <= low else bins - 1 if obs >= high else int((obs - low) / (high - low) * bins)
                   for obs, (low, high) in zip(observation, bounds)]
    return tuple(discretized)

def epsilon_greedy_policy(state, Q, epsilon):
    return env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

def update_Q(Q, state, action, reward, next_state, next_action, alpha, gamma):
    Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    return Q

def store_policy(filename, Q):
    with open(filename, 'wb') as file:
        pickle.dump({'Q_table': Q}, file)

def load_policy(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def store_hyperparameters(filename, hyperparameters):
    with open(filename, 'w') as file:
        json.dump({'hyperparameters': hyperparameters}, file)

def load_hyperparameters(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data['hyperparameters']

def run_simulation(episodes=100, render_mode="human"):
    for i in range(episodes):
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        total_reward = 0

        while True:
            action = epsilon_greedy_policy(state, Q, epsilon=0)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = discretize(observation, n_bins, state_bounds)

            if done or truncated:
                print(f"Simulation episode {i + 1}: Reward: {total_reward}")
                break

# Main execution
if input("Do you want to load the policy and hyperparameters? (yes / just press enter) ").lower() == "yes":
    filename = input("Paste the file name here ")
    loaded_data = load_policy(filename)
    Q = loaded_data['Q_table']

if input("Do you want to run the simulation? (yes/no): ").lower() in ["yes", "y"]:
    env = gym.make("LunarLander-v2", render_mode="human", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    run_simulation()
    env.close()
else:
    # Variables to track the best model
    best_score = -float('inf')
    best_hyperparameters = None
    best_Q = None

    # Stats


    for alpha in alpha_values:
        for gamma in gamma_values:


            for epsilon_decay in epsilon_decay_values:
                print(f"Running for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}")
                epsilon = 1.0
                epsilon_min = 0.01
                Q = np.zeros((n_bins,) * 8 + (nA,))  # Re-initialize Q for each hyperparameter set

                episode_rewards = []
                episode_lengths = []

                for i_episode in range(n_episodes):
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    observation, info = env.reset()
                    state = discretize(observation, n_bins, state_bounds)
                    action = epsilon_greedy_policy(state, Q, epsilon)
                    state_action_reward = deque(maxlen=n + 1)
                    total_reward = 0
                    steps = 0
                    best_reward = 0


                    while True:
                        observation, reward, done, truncated, info = env.step(action)
                        total_reward += reward
                        next_state = discretize(observation, n_bins, state_bounds)
                        next_action = epsilon_greedy_policy(next_state, Q, epsilon)
                        state_action_reward.append((state, action, reward))
                        steps += 1
                        best_reward = max(best_reward, reward)

                        if len(state_action_reward) == n + 1:
                            state_to_update, action_to_update, _ = state_action_reward[0]
                            G = sum(gamma ** i * r for i, (_, _, r) in enumerate(state_action_reward))
                            if not done and not truncated:
                                G += gamma ** n * Q[next_state][next_action]
                            Q = update_Q(Q, state_to_update, action_to_update, G, next_state, next_action, alpha, gamma)

                        if done or truncated:
                            break
                        state, action = next_state, next_action

                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    if np.mean(episode_rewards[-(n_episodes // 10):]) > best_reward:
                        best_reward = np.mean(episode_rewards[-(n_episodes // 10):])
                    # print stats
                    if i_episode % 100 == 0:
                        print(f"Episode: {i_episode + 1}, "
                              f"Average Reward: {np.mean(episode_rewards[-(n_episodes // 10):]):.2f}, "
                              f"Average Length: {np.mean(episode_lengths[-(n_episodes // 10):]):.2f}, "
                              f"Total Reward: {sum(episode_rewards[-(n_episodes // 10):]):.2f}")
                        print(f"Best Reward: {best_reward:.2f}")

                    # Logging and saving best model
                    if total_reward > best_score:
                        best_score = total_reward
                        best_hyperparameters = {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon_decay}
                        best_Q = Q.copy()

                print(
                    f"Finished training for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay} with best score: {best_score}")

    # Save the best policy and hyperparameters
    filename = f"best_policy_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
    store_policy(filename, best_Q)
    store_hyperparameters(filename.replace(".pkl", ".json"), best_hyperparameters)
    print(f"Best policy and hyperparameters saved to {filename}")

playsound('sarsa_new/sound.mp3')
