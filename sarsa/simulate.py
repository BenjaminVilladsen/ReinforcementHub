import gym


from helpers import discretize_state, discretize_state_mountaincar, mountain_car_epsilon_greedy_policy
import numpy as np
from config import settings_lander, settings_car
from playsound import playsound


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
            state = discretize_state(state, [np.linspace(b[0], b[1], settings_lander['num_bins']) for b in settings_lander['state_bounds']])
            action = np.argmax(Q[state])
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        if total_reward >= 200:
            playsound('sound.mp3')
        print(f"Total reward: {total_reward}")


def mountain_car_simulation(Q, bins):
    """
    Run the simulation of the Mountain Car
    :param Q: Q-table
    :param bins: Bins used for state discretization
    :return: None
    """
    env = gym.make('MountainCar-v0', render_mode='human')
    print("Welcome to the Mountain Car simulation!")

    for i in range(40):  # Run the simulation for 40 episodes
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = discretize_state_mountaincar(state, bins=bins)
            action = mountain_car_epsilon_greedy_policy(
                state=state,
                epsilon=0,
                env=env,
                q_table=Q,
            )
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        if state[0] >= 0.5:  # Check if the car reached the goal
            playsound('sound.mp3')  # Play a sound if the goal is reached
        print(f"Total reward: {total_reward}")