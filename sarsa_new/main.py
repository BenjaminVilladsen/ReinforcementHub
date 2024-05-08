from sarsa import sarsa
from config_lunar_lander import state_bounds, num_bins
from helpers import epsilon_greedy_policy, discretize_state, init_q, init_lander_env
from utils import print_text_with_border


def main():
    env = init_lander_env()
    Q = init_q(num_bins=num_bins, env=env)
    print_text_with_border("REINFORCEMENT HUB", px=40, py=2)
    choice = input(
            "What do you want to do? Enter 'train', 'load' or 'grid_search': ").strip().lower()

    if choice == 'train':
        print_text_with_border("TRAIN MODEL", px=40, py=0)
        sarsa(
            epsilon_greedy_policy_fn=epsilon_greedy_policy,
            discretize_fn=discretize_state,
            q_table=Q,
            env=env,
            state_bounds=state_bounds
        )

    elif choice == 'load':
        print_text_with_border("LOAD MODEL")
    else:
        print_text_with_border("INVALID INPUT. EXITING", px=40, py=0)

if __name__ == "__main__":
    main()
