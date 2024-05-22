import gym
import numpy as np
from stable_baselines3 import PPO

# Load the trained PPO model
filename = input("Enter the filename of the trained model: ")
model = PPO.load(filename)

# Create the environment
env = gym.make('CartPole-v1')

# Define evaluation statistics
stats = {
    "mean_reward": 0,
    "wins": 0,
    "std_deviation": 0,
}

episodes = []
num_episodes = 10000  # Define the number of episodes for evaluation
success_threshold = -199  # Define the success threshold (you can adjust this)

for i in range(num_episodes):
    state = env.reset()
    state = state[0]
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action, _ = model.predict(state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    if i % 500 == 0:
        print(f"Episode {i} reward: {total_reward}")
    episodes.append(total_reward)
    stats["mean_reward"] += total_reward
    if total_reward >= success_threshold:
        stats["wins"] += 1

# Calculate mean reward and standard deviation
stats["mean_reward"] /= num_episodes
stats["std_deviation"] = np.std(episodes)

print(stats)




