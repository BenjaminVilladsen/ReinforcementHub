import numpy as np
import gymnasium as gym
import random
from collections import defaultdict

# Initialize the environment
env = gym.make("MountainCar-v0")

# Parameters
num_episodes = 5000
max_steps_per_episode = 200
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.9995
lambda_ = 0.8  # For eligibility traces

# Discretization parameters
num_bins = (20, 20)  # Increased number of bins for finer discretization
bins = [
    np.linspace(-1.2, 0.6, num_bins[0] - 1),
    np.linspace(-0.07, 0.07, num_bins[1] - 1),
]

# Initialize the Q-table with defaultdict
Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Function to discretize state
def discretize_state(state):
    return tuple(
        np.digitize(feature, bins[i])
        for i, feature in enumerate(state)
    )

# Function to choose an action using an epsilon-greedy strategy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(Q_table[state])

# SARSA(λ) training loop
for episode in range(num_episodes):
    state, info = env.reset()
    state = discretize_state(state)
    action = choose_action(state, epsilon)

    # Initialize eligibility traces to zero
    eligibility = defaultdict(lambda: np.zeros(env.action_space.n))

    total_reward = 0

    for step in range(max_steps_per_episode):
        next_state, reward, done, truncated, info = env.step(action)
        next_state = discretize_state(next_state)
        next_action = choose_action(next_state, epsilon)

        # SARSA(λ) update rule
        td_error = reward + discount_factor * Q_table[next_state][next_action] - Q_table[state][action]
        eligibility[state][action] += 1

        # Update all Q-values using eligibility traces
        for s in eligibility:
            for a in range(env.action_space.n):
                Q_table[s][a] += learning_rate * td_error * eligibility[s][a]
                eligibility[s][a] *= discount_factor * lambda_

        state = next_state
        action = next_action
        total_reward += reward

        if done or truncated:
            break

    # Decay epsilon gradually to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()
