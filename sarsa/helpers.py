import numpy as np
import gymnasium as gym


def init_lander_env():
    env = gym.make('LunarLander-v2')

    return env
def epsilon_greedy_policy(state, epsilon, env, q_table):
    return np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(q_table[state])

def epsilon_soft_policy(state, epsilon, env, q_table):
    """
    Epsilon-soft policy: with probability epsilon, select a random action,
    otherwise select the action with the highest Q-value (greedy action).
    """
    if np.random.rand() < epsilon:
        return np.random.choice(env.action_space.n)
    action_probabilities = np.ones(env.action_space.n, dtype=float) * epsilon / env.action_space.n
    best_action = np.argmax(q_table[state])
    action_probabilities[best_action] += (1.0 - epsilon)
    return np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)


def init_q(env, settings):
    # Initialize the Q-table
    q_table_dimensions = [settings['num_bins']] * 6 + [2] * 2 + [env.action_space.n]
    q_table = np.zeros(q_table_dimensions)
    bins = [np.linspace(b[0], b[1], settings['num_bins']) for b in settings['state_bounds']]
    return q_table, bins

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



###################
#MOUNTAINCAR

def init_mountaincar_env():
    env = gym.make('MountainCar-v0')
    return env

def init_q_mountaincar(env, settings):
    # Correct the bins for velocity to match the expected range of the environment
    position_bins = np.linspace(-1.2, 0.6, settings["num_bins"] + 1)  # discretize position with appropriate coverage
    velocity_bins = np.linspace(-0.07, 0.07, settings["num_bins"] + 1)  # CORRECTED: discretize velocity within actual expected range

    # Initialize the Q-table
    q_table_dimensions = [settings["num_bins"]] * 2 + [env.action_space.n]
    q_table = np.zeros(q_table_dimensions)

    # Store bins for later use in discretization (optional, if needed outside init)
    q_table_bins = (position_bins, velocity_bins)

    return q_table, q_table_bins



def discretize_state_mountaincar(state, bins):
    """
    Discretize the continuous state components of the Mountain Car environment.
    """
    if isinstance(state, tuple):
        state_array = state[0]  # Extract the array from the tuple
    else:
        state_array = state  # Directly use the array

        # Discretize each continuous component
    discretized = [int(np.digitize(state_array[i], bins[i]) - 1) for i in range(2)]
    # Append boolean components directly as integers
    return tuple(discretized)


def mountain_car_epsilon_greedy_policy(state, epsilon, env, q_table):
    """
    Select an action for given state using the epsilon-greedy strategy.
    """

    if np.random.rand() < epsilon:
        action = np.random.choice(env.action_space.n)
    else:
        # Ensure that `state` is a tuple of (position_index, velocity_index)
        action = np.argmax(q_table[state[0], state[1], :])  # Access all actions for given state
    return action

def init_cartpole_env():
    env = gym.make('CartPole-v1')
    return env

def init_q_cartpole(env, settings):
    # Initialize the Q-table
    q_table_dimensions = [settings['num_bins']] * 4 + [env.action_space.n]
    q_table = np.zeros(q_table_dimensions)
    bins = [np.linspace(b[0], b[1], settings['num_bins']) for b in settings['state_bounds']]
    return q_table, bins

def discretize_state_cartpole(state, bins):
    """
    Discretize the continuous state components of the CartPole environment.
    """
    if isinstance(state, tuple):
        state = state[0]
    else:
        state = state

    # Discretize each continuous component
    discretized = [int(np.digitize(state[i], bins[i]) - 1) for i in range(4)]
    return tuple(discretized)

def cartpole_epsilon_greedy_policy(state, epsilon, env, q_table):
    """
    Select an action for given state using the epsilon-greedy strategy.
    """

    if np.random.rand() < epsilon:
        action = np.random.choice(env.action_space.n)
    else:
        # Ensure that `state` is a tuple of (position_index, velocity_index)
        action = np.argmax(q_table[state[0], state[1], state[2], state[3], :])  # Access all actions for given state
    return action