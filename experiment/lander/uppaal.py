import numpy as np

from experiment.mountaincar.config import e_config_mountaincar
from stratetrees.models import QTree
import gymnasium as gym

qtree = QTree("/Users/benjamin/Desktop/experiment_lunar.json")

env = gym.make('LunarLander-v2')

stats = {
    "mean_reward": 0,
    "wins": 0,
    "std_deviation": 0,
}

episodes = []



def scale_observation(observation):
    """
    Scale the observation to the range of the Q-table, multiple the first 2 values by 50
    """
    temp = observation.copy()
    temp[0] *= 50
    temp[1] *= 50
    temp[3] *= 50

    return temp

for i in range(e_config_mountaincar['num_episodes']):
    state = env.reset()
    state = state[0]
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action = qtree.predict(scale_observation(state))
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    episodes.append(total_reward)
    stats["mean_reward"] += total_reward
    if total_reward >= e_config_mountaincar['success_threshold']:
        stats["wins"] += 1
    if (i%100 == 0):
        print(f"Episode {i} reward: {total_reward}")

stats["mean_reward"] = np.mean(episodes)
stats["std_deviation"] = np.std(episodes)

print(stats)
