import pickle
import time

import gymnasium as gym
import numpy as np
from collections import deque

# Hyperparameters ranges
alpha_values = [0.5, 0.1, 0.05]
gamma_values = [0.5, 0.9, 0.99]
epsilon_decay_values = [0.5, 1, 0.9]

# Initialize best score
best_score = -float('inf')
best_hyperparameters = None
best_Q = None

n = 10  # Number of steps
n_bins = 10  # Number of bins per state dimension


n_episodes = 5_000

# Environment setup
env = gym.make("LunarLander-v2",
               continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5
               )  # U
nA = env.action_space.n

# Initialize Q-table with a discrete approximation of the state space
Q = np.zeros((n_bins,) * 8 + (nA,))

#store policy and hyperparameters in a file
def storePolicyAndHyperparamsInFile(fileName, Q, hyperparameters):
    data = {
        'Q_table': Q,
        'hyperparameters': hyperparameters
    }
    with open(fileName, 'wb') as file:
        pickle.dump(data, file)

def loadPolicyAndHyperparamsFromFile(fileName):
    """
    Load the policy (Q-table) and hyperparameters from a specified file.

    Parameters:
    - fileName (str): The name of the file from which to load the data.

    Returns:
    - dict: A dictionary containing the Q-table under the key 'Q_table' and
            the hyperparameters under the key 'hyperparameters'.
    """
    with open(fileName, 'rb') as file:
        data = pickle.load(file)
    return data

# Example usage


shouldLoadFromFile = input("Do you want to load the policy and hyperparameters? (yes / just press enter) ")

if shouldLoadFromFile.lower() == "yes":
    loadFileName = input("Paste the file name here ")

    loaded = loadPolicyAndHyperparamsFromFile(loadFileName)
    hyperparameters = loaded['hyperparameters']
    Q = loaded['Q_table']

run_simulation = input("Do you want to run the simulation? (yes/no): ")


# Discretization function
def discretize(observation, bins, bounds):
    discretized = list()
    for i, bound in enumerate(bounds):
        low, high = bound
        if observation[i] <= low:
            idx = 0
        elif observation[i] >= high:
            idx = bins - 1
        else:
            idx = int((observation[i] - low) / (high - low) * bins)
        discretized.append(idx)
    return tuple(discretized)


# Bounds for discretization
state_bounds = [
    (-1.5, 1.5),  # Position X
    (-1.5, 1.5),  # Position Y
    (-5, 5),      # Velocity X
    (-5, 5),      # Velocity Y
    (-3.1415927, 3.1415927),  # Orientation
    (-5, 5),      # Angular Velocity
    (0, 1),       # Left Leg Contact
    (0, 1)        # Right Leg Contact
]


# Epsilon-greedy policy
def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Helper function to update Q-table
def update_Q(Q, state, action, reward, next_state, next_action, alpha, gamma):
    Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    return Q

#Stats
episode_rewards = []
episode_lengths = []


# n step sarsas
best_reward = -99999

for alpha in alpha_values:
    for gamma in gamma_values:
        for epsilon_decay in epsilon_decay_values:
            print(f"Running for alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}")
            epsilon_start = 1.0
            epsilon_min = 0.01
            epsilon = epsilon_start

            if (shouldLoadFromFile != "yes"):
                for i_episode in range(n_episodes):  # Number of episodes
                    epsilon = max(epsilon * epsilon_decay, epsilon_min)
                    #episode stats
                    total_reward = 0
                    steps = 0
                    best_reward = 0

                    observation, info = env.reset()
                    state = discretize(observation, n_bins, state_bounds)
                    action = epsilon_greedy_policy(state, Q, epsilon)
                    state_action_reward = deque(maxlen=n + 1)

                    action_avg_reward = 0

                    while True:
                        observation, reward, done, truncated, info = env.step(action)
                        total_reward += reward
                        best_reward = max(best_reward, reward)
                        steps += 1

                        next_state = discretize(observation, n_bins, state_bounds)
                        next_action = epsilon_greedy_policy(next_state, Q, epsilon)
                        state_action_reward.append((state, action, reward))

                        if len(state_action_reward) >= n + 1:
                            state_to_update, action_to_update, _ = state_action_reward[0]
                            G = sum([gamma ** i * r for i, (_, _, r) in enumerate(state_action_reward)])
                            if not done and not truncated:
                                G += gamma ** n * Q[next_state][next_action]
                            Q = update_Q(Q, state_to_update, action_to_update, G, next_state, next_action, alpha, gamma)

                        if done or truncated:
                            break
                        state = next_state
                        action = next_action

                    episode_rewards.append(total_reward)
                    episode_lengths.append(steps)
                    if np.mean(episode_rewards[-(n_episodes // 10):]) > best_reward:
                        best_reward = np.mean(episode_rewards[-(n_episodes // 10):])

                    #print stats
                    if (i_episode + 1) % (n_episodes // 10) == 0:
                        print(f"Episode: {i_episode + 1}, "
                              f"Average Reward: {np.mean(episode_rewards[-(n_episodes // 10):]):.2f}, "
                              f"Average Length: {np.mean(episode_lengths[-(n_episodes // 10):]):.2f}, "
                              f"Total Reward: {sum(episode_rewards[-(n_episodes // 10):]):.2f}")
                        print(f"Best Reward: {best_reward:.2f}")


                env.close()
                current_score = np.mean(episode_rewards[-100:])

                # Update best score and hyperparameters if current model is better
                if current_score > best_score:
                    best_score = current_score
                    best_hyperparameters = {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon_decay}
                    best_Q = Q.copy()  # Make sure to copy the Q-table, not just reference it

#file name with best cost in policy and time and date
#current time
currentTimeStr = time.strftime("%Y%m%d-%H")
fileName = f"Policy.txt"

#as dict
hyperparameters = {
    'alpha': alpha,
    'gamma': gamma,
    'n': n,
    'n_bins': n_bins,
    'epsilon_start': epsilon_start,
    'epsilon_decay': epsilon_decay,
    'epsilon_min': epsilon_min,
    'n_episodes': n_episodes
}

storePolicyAndHyperparamsInFile(fileName, best_Q, best_hyperparameters)


# ---------------------------------------------------
# -----------------SIMULATION------------------------
# ---------------------------------------------------

if run_simulation.lower() != "y" and run_simulation.lower() != "yes":
    exit()
env = gym.make("LunarLander-v2", render_mode="human",
               continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5
               )
for i_simulation in range(100):
    observation, info = env.reset()
    state = discretize(observation, n_bins, state_bounds)
    total_reward = 0

    while True:
        action = epsilon_greedy_policy(state, Q, epsilon=0)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = discretize(observation, n_bins, state_bounds)

        if done or truncated:
            print(f"Simulation episode {i_simulation + 1}: Reward: {total_reward}")
            break

env.close()



