import numpy as np

from experiment.cartpole.config import e_config_pole
from stratetrees.models import QTree
import gymnasium as gym

qtree = QTree("/Users/benjamin/Desktop/experiment_cartpole.json")


env = gym.make('CartPole-v1')

stats = {
    "mean_reward": 0,
    "wins": 0,
    "std_deviation": 0,
}

episodes = []



def invertAction(action):
    if action == 1:
        return 0
    return 1


for i in range(e_config_pole['num_episodes']):
    state = env.reset()
    state = state[0]
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        action = qtree.predict(state)
        state, reward, done, truncated, _ = env.step(invertAction(action))
        total_reward += reward

    episodes.append(total_reward)
    stats["mean_reward"] += total_reward
    if total_reward >= e_config_pole['success_threshold']:
        stats["wins"] += 1
    if(i % 100 == 0):
        print(f"Episode {i} reward: {total_reward}")


stats["mean_reward"] = np.mean(episodes)
stats["std_deviation"] = np.std(episodes)

print(stats)
