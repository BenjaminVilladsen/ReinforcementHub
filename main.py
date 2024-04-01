import gymnasium as gym
import numpy as np
from collections import deque

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.4  # Discount factor
epsilon = 0.5  # Epsilon for the ε-greedy policy
n = 10  # Number of steps
n_bins = 10  # Number of bins per state dimension

# Environment setup
env = gym.make("LunarLander-v2",
               continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5
               )  # U
nA = env.action_space.n

# Initialize Q-table with a discrete approximation of the state space
Q = np.zeros((n_bins,) * 8 + (nA,))


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
            # Scale the observation to the range [0, n_bins - 1]
            idx = int((observation[i] - low) / (high - low) * bins)
        discretized.append(idx)
    return tuple(discretized)


# Bounds for discretization
# These need to be chosen based on observed min/max values of each dimension
state_bounds = [(-1, 1)] * 8  # Placeholder bounds, replace with actual observed bounds


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


# n-step Sarsa loop
for i_episode in range(999999):  # Number of episodes
    win_counter = 0
    total_rounds = 0
    observation, info = env.reset()
    state = discretize(observation, n_bins, state_bounds)
    action = epsilon_greedy_policy(state, Q, epsilon)
    state_action_reward = deque(maxlen=n + 1)

    for t in range(999999999):  # Limit the number of timesteps per episode
        observation, reward, done, truncated, info = env.step(action)
        next_state = discretize(observation, n_bins, state_bounds)
        next_action = epsilon_greedy_policy(next_state, Q, epsilon)
        state_action_reward.append((state, action, reward))

        if t >= n:
            state_to_update, action_to_update, _ = state_action_reward[0]
            G = sum([gamma ** i * r for i, (_, _, r) in enumerate(state_action_reward)])
            if not done and not truncated:
                G += gamma ** n * Q[next_state][next_action]
            Q = update_Q(Q, state_to_update, action_to_update, G, next_state, next_action, alpha, gamma)


        if done or truncated:
            break
        state = next_state
        action = next_action
    if (i_episode % 1000 == 0):
        print("Episode", i_episode, "")

env.close()


# ---------------------------------------------------
# -----------------SIMULATION------------------------
# ---------------------------------------------------
env = gym.make("LunarLander-v2", render_mode="human",
               continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5
               )  # Use render_mode="human" to visualize
for i_simulation in range(100):  # Run 5 simulation episodes
    observation, info = env.reset()
    state = discretize(observation, n_bins, state_bounds)
    total_reward = 0

    while True:
        action = epsilon_greedy_policy(state, Q, epsilon=0)  # Now epsilon=0 for exploitation
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        state = discretize(observation, n_bins, state_bounds)

        if done or truncated:
            print(f"Simulation episode {i_simulation + 1}: Total Reward: {total_reward}")
            break

env.close()
