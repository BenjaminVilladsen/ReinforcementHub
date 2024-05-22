# LunarLander_statebounds
lander_bounds = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.1415927, 3.1415927), (-5, 5)]
car_bounds = [(-1.2, 0.6), (-0.07, 0.07)]  # Corrected bounds
pole_bounds = [(-2.4, 2.4), (-float("inf"), float("inf")), (-0.2095, 0.2095), (-float("inf"), float("inf"))]


e_config_pole = {
    "num_episodes" : 10_000,
    "epsilon" : 0,
    "success_threshold": 500,
    "state_bounds": pole_bounds
}
