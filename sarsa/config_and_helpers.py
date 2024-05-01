import numpy as np
import gymnasium as gym

# Hyperparameters for tuning the learning process
alpha_values = [0.5]  # Learning rate values to experiment with
gamma_values = [0.5]  # Discount factor values for future rewards
epsilon_decay_values = [0.9]  # Epsilon decay rates for the epsilon-greedy policy

# Simulation parameters
n_episodes = 2000  # Number of episodes to run the simulation
n_bins = 10  # Number of bins for discretizing each state dimension
n = 1000  # Length of the reward sequence for n-step Q-learning

# Environment setup
env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
               turbulence_power=1.5)
nA = env.action_space.n  # Number of possible actions
Q = np.zeros((n_bins,) * 8 + (nA,))  # Initialize the Q-table with all zeros

# State bounds for discretization of continuous state space
state_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5), (0, 1), (0, 1)]

def discretize(observation, bins, bounds):
    discretized = [0 if obs <= low else bins - 1 if obs >= high else int((obs - low) / (high - low) * bins)
                   for obs, (low, high) in zip(observation, bounds)]
    return tuple(discretized)

def epsilon_greedy_policy(state, Q, epsilon):
    return env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

def update_Q(Q, state, action, reward, next_state, next_action, alpha, gamma):
    Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    return Q
