import time

from sarsa import sarsa
from config_lunar_lander import state_bounds, num_bins
from helpers import epsilon_greedy_policy, discretize_state, init_q, init_lander_env
from file_handling import store_policy, load_policy
from simulate import lander_simulation
from utils import print_text_with_border
import numpy as np


def main():
    env = init_lander_env()
    Q = init_q(num_bins=num_bins, env=env)
    print_text_with_border("REINFORCEMENT HUB", px=40, py=2)
    choice = input(
            "What do you want to do? Enter 'train', 'load' or 'grid_search': ").strip().lower()

    if choice == 'train':
        print_text_with_border("TRAIN MODEL", px=40, py=0)
        Q, epsiode_rewards = sarsa(
            epsilon_greedy_policy_fn=epsilon_greedy_policy,
            discretize_fn=discretize_state,
            q_table=Q,
            env=env,
            state_bounds=state_bounds
        )

        filename = f"best_policy_{np.mean(epsiode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q)


    elif choice == 'load':
        print_text_with_border("LOAD MODEL")
        filename = input("Enter the filename of the saved model: ")
        loaded_data = load_policy(filename)
        Q[:] = loaded_data['Q_table']
    else:
        print_text_with_border("INVALID INPUT. EXITING", px=40, py=0)
        return

    simulate = input("Do you want to simulate the model? (y/n): ").strip().lower()
    if simulate == 'y':
        lander_simulation(Q)
    else:
        print_text_with_border("EXITING", px=40, py=0)



if __name__ == "__main__":
    main()
