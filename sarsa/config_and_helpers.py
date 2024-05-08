import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

############################################################
#            HYPERPARAMETERS / TRAINING OPTIONS            #
############################################################

alpha = 0.3  # Learning rate values to experiment with
gamma = 0.9  # Discount factor values for future rewards high -> more exploration, low -> less exploration
n_episodes = 1000  # Number of episodes to train the simulation
n_bins = 5 # Number of bins for discretizing each state dimension
n = 3
init_epsilon = 0.9
epsilon_decay = 0.99999
min_epsilon = 0.2
log_interval = 100
max_time_steps = 300



# State bounds for discretization
state_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5), (0, 1), (0, 1)]


def initializeQAndEnv():
    """
    :return: env, nA, Q
    """
    env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
                   turbulence_power=1.5)  # Environment setup

    nA = env.action_space.n  # Number of possible actions
    Q = np.zeros((n_bins,) * 8 + (nA,))  # Initialize the Q-table with all zeros
    return Q, env, nA

def initializeQAndEnv():
    """
    Initialize Q-table and environment.

    Returns:
        Q (np.array): Initialized Q-table.
        env: Gym environment.
        nA (int): Number of actions.
    """
    env = gym.make("LunarLander-v2")
    nA = env.action_space.n
    n_bins = 5
    n_states = [n_bins] * env.observation_space.shape[0]
    Q = np.zeros(n_states + [nA])
    return Q, env, nA


#write a discretizie function
def discretize(observation, n_bins, bounds):
    """
    Discretize the continuous state into a discrete state.

    Parameters:
        observation (np.array): The continuous state.
        n_bins (int): Number of bins to use.
        bounds (list): List of tuples with the lower and upper bounds for each dimension.

    Returns:
        tuple: The discretized state.
    """
    state = []
    for i, x in enumerate(observation):
        l, u = bounds[i]
        state.append(int(np.digitize(x, np.linspace(l, u, n_bins))))
    return tuple(state)




def epsilon_greedy_policy(state, Q_policy, epsilon, env):
    return env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q_policy[state])


def update_Q_n_step(Q, state, action, G, next_state, next_action, alpha, gamma, done):
    current = Q[state][action]
    future = Q[next_state][next_action] if not done else 0
    Q[state][action] += alpha * (G + (gamma ** n) * future - current)
    return Q


def print_stats_header(episode_num):
    print("-" * 80)
    print("| {:^20} | {:^30} | {:^10} |".format("Statistic", f"Episode {episode_num}", "All Episodes"))
    print("|" + "-" * 22 + "|" + "-" * 32 + "|" + "-" * 12 + "|")


def print_stats_bottom():
    print("-" * 80)


def print_stats_line(statistic, episode_val, all_val):
    episode_val_str = "{:.2f}".format(episode_val) if isinstance(episode_val, float) else str(episode_val)
    all_val_str = "{:.2f}".format(all_val) if isinstance(all_val, float) else str(all_val)
    max_statistic_width = max(len(statistic), 20)  # Minimum width of 50 for the "Statistic" column
    max_episode_width = max(len(episode_val_str), 30)
    max_all_width = max(len(all_val_str), 10)

    print("| {:<{}} | {:^{}} | {:^{}} |".format(statistic, max_statistic_width, episode_val_str, max_episode_width,
                                                all_val_str, max_all_width))


def print_stats(episode_num=0, e_avg_reward=0, t_avg_reward=0, e_best=0, t_best=0, e_worst=0, t_worst=0,
                e_above_200_count=0, t_above_200_count=0, e_above_100_count=0, t_above_100_count=0):
    print_stats_header(episode_num)
    print_stats_line("avg_reward", e_avg_reward, t_avg_reward)
    print_stats_line("best_reward", e_best, t_best)
    print_stats_line("worst_reward", e_worst, t_worst)
    print_stats_line("reward > 200 count", e_above_200_count, t_above_200_count)
    print_stats_line("reward > 100 count", e_above_100_count, t_above_100_count)
    print_stats_bottom()


def plot_rewards(episode_rewards, window=100):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode reward')
    rolling_mean = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    plt.plot(np.arange(window - 1, len(episode_rewards)), rolling_mean,
             label=f'Rolling average (window size = {window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()
