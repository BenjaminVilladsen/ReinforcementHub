
# LunarLander_statebounds
state_bounds_lander = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5)]
state_bounds_car = [(-1.2, 0.6), (-0.07, 0.07)]  # Corrected bounds
state_bounds_cartpole = [(-2.4, 2.4), (-float("inf"), float("inf")), (-0.2095, 0.2095), (-float("inf"), float("inf"))]


settings_lander = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 1,  # You might also consider an epsilon decay strategy here
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.01,
    "num_bins": 20,
    "num_episodes": 20_000,
    "log_interval": 1_000,
    "state_bounds": state_bounds_lander,
    "convergence_threshold": 0.01,
    "success_threshold": 200
}
settings_car = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 1,  # You might also consider an epsilon decay strategy here
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.01,
    "num_bins": 20,
    "num_episodes": 10_000,
    "log_interval": 500,
    "state_bounds": state_bounds_car,
    "convergence_threshold": 0.01,
    "success_threshold": -150
}

settings_cartpole = {
    "alpha": 0.05,
    "gamma": 0.95,
    "epsilon": 1,  # You might also consider an epsilon decay strategy here
    "epsilon_decay": 0.99995,
    "epsilon_min": 0.01,
    "num_bins": 50,
    "num_episodes": 100_000,
    "log_interval": 1_000,
    "state_bounds": state_bounds_cartpole,
    "convergence_threshold": 0.005,
    "success_threshold": 200  # Adjust based on the specific CartPole version
}


