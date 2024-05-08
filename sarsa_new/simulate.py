import gym

from config_lunar_lander import num_bins, state_bounds
from helpers import discretize_state
import numpy as np

def lander_simulation(Q):
    """
    Run the simulation of the lunar lander
    :param Q: Q-table
    :param env: Environment
    :return: None
    """
    env = gym.make('LunarLander-v2', render_mode='human')
    print("Welcome to the Lunar Lander simulation!")


    for i in range(40):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = discretize_state(state, [np.linspace(b[0], b[1], num_bins) for b in state_bounds])
            action = np.argmax(Q[state])
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Total reward: {total_reward}")