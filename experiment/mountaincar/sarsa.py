import gym
import numpy as np

from config import e_config_mountaincar
from experiment.imports_are_not_working.copied import load_policy, mountain_car_epsilon_greedy_policy, \
    discretize_state_mountaincar, init_q_mountaincar

filename = input("enter filename: ")
q_table, settings = load_policy(filename)

env = gym.make('MountainCar-v0')

stats = {
    "mean_reward": 0,
    "wins": 0,
    "std_deviation": 0,
}

episodes = []

_, bins = init_q_mountaincar(env, settings)

for i in range(e_config_mountaincar['num_episodes']):
    state = env.reset()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        state = discretize_state_mountaincar(state, bins=bins)
        action = mountain_car_epsilon_greedy_policy(
            state=state,
            epsilon=0,
            env=env,
            q_table=q_table,
        )
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    episodes.append(total_reward)
    stats["mean_reward"] += total_reward
    if total_reward >= e_config_mountaincar['success_threshold']:
        stats["wins"] += 1


stats["mean_reward"] = np.mean(episodes)
stats["std_deviation"] = np.std(episodes)

print(stats)
