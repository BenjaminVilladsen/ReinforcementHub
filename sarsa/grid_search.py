import itertools

from training import train


def grid_search(params_grid, **fixed_params):
    """
    Perform a grid search over the specified parameters.

    Parameters:
    - params_grid (dict): A dictionary where keys are parameter names and values are lists of values to test.
    - **fixed_params: Additional fixed parameters for the training function.

    Returns:
    - best_params (dict): The best combination of parameters found during the search.
    - best_avg_reward (float): The average reward achieved with the best parameter combination.
    """

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*params_grid.values()))

    best_avg_reward = -float('inf')
    best_params = None
    best_Q = None
    i = 0

    for params in param_combinations:
        i = i +1
        # Create a dictionary with the current combination of parameters
        current_params = dict(zip(params_grid.keys(), params))
        # Add fixed parameters
        current_params.update(fixed_params)

        print(f"{i}/{len(param_combinations)}  Testing parameters: {current_params}")

        # Call the training function with the current parameters
        Q, avg_reward = train(**current_params, verbose=False, storeFile=False)

        # Update best parameters if current average reward is better
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_params = current_params
            best_Q = Q


    print(f"Grid search finished. Best parameters: {best_params}, Best average reward: {best_avg_reward}")

    return best_Q, best_params, best_avg_reward

# Define the grid of hyperparameters to search over
params_grid = {
    'alpha': [0.01, 0.3],
    'gamma': [0.6, 0.3],
    'n': [3, 10],
    'epsilon_decay': [1, 0.7],
    'min_epsilon': [0.01, 0.2]
}

# Fixed parameters
fixed_params = {
    'n_episodes': 1000,
    'log_interval': 200,
    'max_time_steps': 2000,
    'init_epsilon': 0.9,
}


