import numpy as np
import gym


def init_lander_env():
    env = gym.make('LunarLander-v2')
    return env
def epsilon_greedy_policy(state, epsilon, env, q_table):
    return np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(q_table[state])


def init_q(num_bins, env):
    # Initialize the Q-table
    q_table_dimensions = [num_bins] * 6 + [2] * 2 + [env.action_space.n]
    q_table = np.zeros(q_table_dimensions)
    return q_table

def discretize_state(state, bins):
    """
    The env.reset returns tuple, env.step does not
    :param state:
    :return:
    """
    # Check if the state is in a tuple format and extract the array part
    if isinstance(state, tuple):
        state_array = state[0]  # Extract the array from the tuple
    else:
        state_array = state  # Directly use the array

    # Discretize each continuous component
    discretized = [int(np.digitize(state_array[i], bins[i]) - 1) for i in range(6)]
    # Append boolean components directly as integers
    discretized.extend([int(state_array[i]) for i in range(6, 8)])
    return tuple(discretized)
