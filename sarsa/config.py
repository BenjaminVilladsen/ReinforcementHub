
# LunarLander_statebounds
state_bounds_lander = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5)]
state_bounds_car = [(-1.2, 0.6), (-0.07, 0.07)]  # Corrected bounds
state_bounds_cartpole = [(-2.4, 2.4), (-float("inf"), float("inf")), (-0.2095, 0.2095), (-float("inf"), float("inf"))]


settings_lander = {
    "alpha": 0.003,
    "gamma": 0.9,
    "epsilon": 0.1,
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.01,
    "num_bins": 20,
    "num_episodes": 100_000,
    "log_interval": 1000,
    "state_bounds": state_bounds_lander,
    "convergence_threshold": 0.001,
    "success_threshold": 200,
    "convergence_count_limit": 1000,
}
settings_car = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 1,  # You might also consider an epsilon decay strategy here
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.01,
    "num_bins": 1000,
    "num_episodes": 100_000,
    "log_interval": 1000,
    "state_bounds": state_bounds_car,
    "convergence_threshold": 0.001,
    "success_threshold": -150,
    "convergence_count_limit": 30000,
}

settings_cartpole = {
    "alpha": 0.009,  # Adjusted to be lower, similar to the learning rate used in PPO
    "gamma": 0.98,  # Same as the gamma in PPO
    "epsilon": 0.6,  # Initial exploration rate, might need decay
    "epsilon_decay": 0.99995,  # Adjusted to have a slower decay over a large number of episodes
    "epsilon_min": 0.1,  # Minimum exploration rate
    "num_bins": 1100,  # Discretization for state space, remains the same
    "num_episodes": 100_000,  # Same number of timesteps, converted to episodes
    "log_interval": 1000,  # Logging interval remains the same
    "state_bounds": state_bounds_cartpole,  # Boundaries for state space
    "convergence_threshold": 0.00001,  # Convergence threshold for stopping
    "success_threshold": 500,  # Maximum score for CartPole-v1
    "convergence_count_limit": 1_000_000,  # Limit for convergence count
}



