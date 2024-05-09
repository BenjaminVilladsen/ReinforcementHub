
# LunarLander_statebounds
state_bounds_lander = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5)]
state_bounds_car = [(-1.2, 0.6), (-0.07, 0.07)]  # Corrected bounds

settings_lander = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.1,
    "num_bins": 20,
    "num_episodes": 1000,
    "log_interval": 500,
    "state_bounds": state_bounds_lander
}

settings_car = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 0.1,
    "num_bins": 20,
    "num_episodes": 10_000,
    "log_interval": 500,
    "state_bounds": state_bounds_car
}

