import time
from grid_search import grid_search, params_grid, fixed_params
from training import train
from simulation import run_simulation
from file_operations import load_policy, store_policy
from config_and_helpers import n, alpha, gamma, n_episodes, log_interval, max_time_steps, \
    init_epsilon, min_epsilon, epsilon_decay, initializeQAndEnv


def main():

    Q_Sim, env, nA = initializeQAndEnv()
    print("Welcome to the Lunar Lander simulation!")
    choice = input(
        "Do you want to train a new model or load an existing one? Enter 'train', 'load' or 'grid_search': ").strip().lower()
    if choice == 'train':
        Q_Sim, _ = train(
            n=n,
            alpha=alpha,
            gamma=gamma,
            n_episodes=n_episodes,
            log_interval=log_interval,
            max_time_steps=max_time_steps,
            init_epsilon=init_epsilon,
            min_epsilon=min_epsilon,
            epsilon_decay=epsilon_decay)
    elif choice == 'load':
        filename = input("Please enter the filename of the saved model: ")
        loaded_data = load_policy(filename)
        Q_Sim[:] = loaded_data['Q_table']  # Update the Q-table with the loaded policy
    elif choice == 'grid_search':
        bestQ, best_params, best_avg_reward = grid_search(params_grid, **fixed_params)
        print("Finished grid search")
        print("Best params: ", best_params)
        print("Best avg_reward: ", best_avg_reward)
        filename = f"grid_search_best_{best_avg_reward}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, bestQ)
        store_policy(filename.replace(".pkl", ".json"), best_params)
        Q_Sim[:] = bestQ


    else:
        print("Invalid input. Exiting.")
        return

    run_simulation(policy_Q=Q_Sim) # run the simulation

if __name__ == "__main__":
    main()
