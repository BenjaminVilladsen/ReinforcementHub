import numpy as np
import gymnasium as gym

############################################################
#            HYPERPARAMETERS / TRAINING OPTIONS            #
############################################################

alpha = 0.01  # Learning rate values to experiment with
gamma = 0.9  # Discount factor values for future rewards high -> more exploration, low -> less exploration
n_episodes = 3000  # Number of episodes to train the simulation
n_bins = 10  # Number of bins for discretizing each state dimension
n = 6
init_epsilon = 1
epsilon_decay = 0.999
min_epsilon = 0.1
log_interval = 200
max_time_steps = 5_000


env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
               turbulence_power=1.5) # Environment setup


nA = env.action_space.n  # Number of possible actions
Q = np.zeros((n_bins,) * 8 + (nA,))  # Initialize the Q-table with all zeros

# State bounds for discretization
state_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5), (0, 1), (0, 1)]

def discretize(observation, bins, bounds):
    discretized = [0 if obs <= low else bins - 1 if obs >= high else int((obs - low) / (high - low) * bins)
                   for obs, (low, high) in zip(observation, bounds)]
    return tuple(discretized)

def epsilon_greedy_policy(state, Q, epsilon):
    return env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Q[state])

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

    print("| {:<{}} | {:^{}} | {:^{}} |".format(statistic, max_statistic_width, episode_val_str, max_episode_width, all_val_str, max_all_width))

def print_stats(episode_num=0, e_avg_reward=0, t_avg_reward=0, e_best=0, t_best=0, e_worst=0, t_worst=0, e_above_200_count=0, t_above_200_count=0, e_above_100_count=0, t_above_100_count=0):
    print_stats_header(episode_num)
    print_stats_line("avg_reward", e_avg_reward, t_avg_reward)
    print_stats_line("best_reward", e_best, t_best)
    print_stats_line("worst_reward", e_worst, t_worst)
    print_stats_line("reward > 200 count", e_above_200_count, t_above_200_count)
    print_stats_line("reward > 100 count", e_above_100_count, t_above_100_count)
    print_stats_bottom()
