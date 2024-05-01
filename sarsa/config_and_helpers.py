import numpy as np
import gymnasium as gym

############################################################
#            HYPERPARAMETERS / TRAINING OPTIONS            #
############################################################

alpha_values = [0.1]  # Learning rate values to experiment with
gamma_values = [0.99]  # Discount factor values for future rewards
epsilon_decay_values = [0.9]  # Epsilon decay rates for the epsilon-greedy policy
n_episodes = 20_000  # Number of episodes to train the simulation
n_bins = 10  # Number of bins for discretizing each state dimension
n = 6


env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
               turbulence_power=1.5) # Environment setup


nA = env.action_space.n  # Number of possible actions
Q = np.zeros((n_bins,) * 8 + (nA,))  # Initialize the Q-table with all zeros

# State bounds for discretization
state_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5), (0, 1), (0, 1)]

def discretize(observation, bins, bounds):
    discretized = [0 if obs <= low else bins - 1 if obs >= high else int((obs - low) / (high - low) * bins)
                   for obs, (low, high) in zip(observation, bounds)]
    return tuple(discretized)

def epsilon_greedy_policy(state, Q, epsilon):
    return env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

def update_Q_n_step(Q, state, action, G, next_state, next_action, alpha, gamma, done):
    current = Q[state][action]
    future = Q[next_state][next_action] if not done else 0
    Q[state][action] += alpha * (G + (gamma ** n) * future - current)
    return Q
