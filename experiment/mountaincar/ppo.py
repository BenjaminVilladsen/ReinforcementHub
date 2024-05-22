import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import os

# Load the trained PPO model
filename = input("Enter the filename of the trained model: ")
model = PPO.load(filename)

# Create the vectorized environment and normalize it
env = DummyVecEnv([lambda: gym.make('MountainCar-v0')])
env = VecNormalize.load(os.path.join(os.path.dirname(filename), 'vec_normalize.pkl'), env)

# Don't update the statistics in evaluation
env.training = False
env.norm_reward = False

# Define evaluation statistics
stats = {
    "mean_reward": 0,
    "wins": 0,
    "std_deviation": 0,
    "best_reward": -float('inf')
}

episodes = []
num_episodes = 10000  # Define the number of episodes for evaluation
success_threshold = -199  # Define the success threshold

for i in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    if i % 500 == 0:
        print(f"Episode {i} reward: {total_reward}")
    if total_reward > stats["best_reward"]:
        stats["best_reward"] = total_reward
        print(f"New best reward: {stats['best_reward']}")
    episodes.append(total_reward)
    stats["mean_reward"] += total_reward
    if total_reward >= success_threshold:
        stats["wins"] += 1

# Calculate mean reward and standard deviation
stats["mean_reward"] /= num_episodes
stats["std_deviation"] = np.std(episodes)

print(stats)
