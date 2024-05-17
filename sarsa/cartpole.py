import time

from sarsa import sarsa
from config import settings_cartpole
from helpers import cartpole_epsilon_greedy_policy, discretize_state_cartpole, init_cartpole_env, \
    init_q_cartpole, epsilon_soft_policy
from file_handling import store_policy, load_policy
from simulate import cartpole_simulation
from utils import print_text_with_border, plot_rewards, print_episode_stats, print_stats_lander
import numpy as np


def main():
    env = init_cartpole_env()
    Q, bins = init_q_cartpole(env=env, settings=settings_cartpole)
    print_text_with_border("REINFORCEMENT HUB - CARTPOLE", px=40, py=2)
    choice = input(
            "What do you want to do? train ('t'), load ('l'), load_and_train ('lt') or grid_search('gs'): ").strip().lower()

    if choice == 't':
        print_text_with_border("TRAIN MODEL", px=40, py=0)
        Q, episode_rewards = sarsa(
            epsilon_greedy_policy_fn=epsilon_soft_policy,
            discretize_fn=discretize_state_cartpole,
            q_table=Q,
            env=env,
            settings=settings_cartpole,
            bins=bins,
            print_fn=print_stats_lander
        )

        filename = f"cp_best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q, settings_cartpole)

        #plot rewards
        plot_rewards(episode_rewards)

    elif choice == 'lt':
        print_text_with_border("LOAD AND TRAIN", px=40, py=0)
        filename = input("Enter the filename of the saved model (with '.pkl'): ")
        loaded_q, loaded_settings = load_policy(filename)
        episodes_num = int(input("Enter the number of episodes to train: "))
        loaded_settings['num_episodes'] = episodes_num
        Q[:] = loaded_q
        Q_trained, episode_rewards = sarsa(
            epsilon_greedy_policy_fn=epsilon_soft_policy,
            discretize_fn=discretize_state_cartpole,
            q_table=Q,
            env=env,
            settings=settings_cartpole,
            bins=bins,
            print_fn=print_stats_lander
        )
        Q = Q_trained
        filename = f"cp_updated_best_policy_{np.mean(episode_rewards)}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
        store_policy(filename, Q, settings_cartpole)
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
        cartpole_simulation(Q, bins=bins)
    else:
        print_text_with_border("EXITING", px=40, py=0)



if __name__ == "__main__":
    main()
