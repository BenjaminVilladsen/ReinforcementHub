import time

from sarsa import sarsa
from config import settings_lander
from helpers import epsilon_greedy_policy, discretize_state, init_q, init_lander_env
from file_handling import store_policy, load_policy
from simulate import lander_simulation
from utils import print_text_with_border, plot_rewards
import numpy as np


def main():
    env = init_lander_env()
    Q = init_q(num_bins=settings_lander['num_bins'], env=env)
    print_text_with_border("REINFORCEMENT HUB", px=40, py=2)
    choice = input(
            "What do you want to do? train ('t'), load ('l'), load_and_train ('lt') or grid_search('gs'): ").strip().lower()

    if choice == 't':
        print_text_with_border("TRAIN MODEL", px=40, py=0)
        Q, episode_rewards = sarsa(
            epsilon_greedy_policy_fn=epsilon_greedy_policy,
            discretize_fn=discretize_state,
            q_table=Q,
            env=env,
            settings=settings_lander
        )

        filename = f"best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q, settings_lander)

        #plot rewards
        plot_rewards(episode_rewards)

    elif choice == 'lt':
        print_text_with_border("LOAD AND TRAIN", px=40, py=0)
        filename = input("Enter the filename of the saved model (with '.pkl'): ")
        loaded_q, loaded_settings = load_policy(filename)
        episodes_num = int(input("Enter the number of episodes to train: "))

        Q[:] = loaded_q
        Q_trained, episode_rewards = sarsa(
            epsilon_greedy_policy_fn=epsilon_greedy_policy,
            discretize_fn=discretize_state,
            q_table=Q,
            env=env,
            settings=loaded_settings
        )
        Q = Q_trained
        filename = f"updated_best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q, settings_lander)
        plot_rewards(episode_rewards)

    elif choice == 'l':
        print_text_with_border("LOAD MODEL", px=40, py=0)
        filename = input("Enter the filename of the saved model: ")
        loaded_q, loaded_settings = load_policy(filename)
        print("Model loaded successfully!")
        print("Hyperparams: ", loaded_settings)
        Q[:] = loaded_q
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
