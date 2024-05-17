import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
def print_text_with_border(text: str, px=10, py=1) -> None:
    #half of px and py but as integers
    px_half = int(px / 2)
    py_half = int(py / 2)
    #print the top border
    print(f"+{'-' * (len(text) + px)}+")
    for _ in range(py_half):
        print(f"|{' ' * (len(text) + px)}|")

    #print the text with side borders
    print(f"|{' ' * px_half}{text}{' ' * px_half}|")

    for _ in range(py_half):
        print(f"|{' ' * (len(text) + px)}|")

    #print the bottom border
    print(f"+{'-' * (len(text) + px)}+")


def print_episode_stats(episode_rewards, i_episode, episode_span=100):
    """
    dict of this type:
    {
    'num_episode': i_episode + 1,
    'avg_reward': total_reward,
    }
    :return:
    """
    avg_reward = np.mean(episode_rewards[-episode_span:])
    num_episodes_passed = len([r for r in episode_rewards[-episode_span:] if r >= 200])
    num_episodes_failed = episode_span - num_episodes_passed
    success_rate = f"{num_episodes_passed / episode_span * 100}"

    best_reward = f"{np.max(episode_rewards[-episode_span:])}"
    worst_reward = f"{np.min(episode_rewards[-episode_span:])}"

    console = Console()
    table = Table(title=f"Episode {i_episode}", title_style="bold blue", show_header=True, header_style="bold magenta")
    table.add_column("Average", style="dim", width=12)
    table.add_column("Passed")
    table.add_column("Failed")
    table.add_column("Percent")
    table.add_column("Best")
    table.add_column("Worst")
    table.add_row(str(avg_reward), str(num_episodes_passed), str(num_episodes_failed), str(success_rate), str(best_reward), str(worst_reward))
    console.print(table)


def print_stats_lander(epsiode_rewards, title, settings, convergence_count, success_count, time_elapsed):
    console = Console()
    table = Table(title=f"Episode {title}", title_style="bold blue", show_header=True, header_style="bold magenta")
    table.add_column("Context", style="dim")
    table.add_column("Time elapsed", style="dim")
    table.add_column("Reward mean", style="dim")
    table.add_column("+- std deviation", style="dim")
    table.add_column("Convergence count", style="dim")
    table.add_column("Success rate", style="dim")
    table.add_column("Best", style="dim")
    table.add_column("Worst", style="dim")


    reward_mean = np.mean(epsiode_rewards)
    plus_minus_std_deviation = np.std(epsiode_rewards)
    best = np.max(epsiode_rewards)
    worst = np.min(epsiode_rewards)

    # recent epsiodes
    interval = settings['log_interval']
    r_reward_mean = np.mean(epsiode_rewards[-interval:])
    r_plus_minus_std_deviation = np.std(epsiode_rewards[-interval:])
    r_best = np.max(epsiode_rewards[-interval:])
    r_worst = np.min(epsiode_rewards[-interval:])
    r_success_count = len([r for r in epsiode_rewards[-interval:] if r >= 200])

    table.add_row(
        "Overall",
        f"{time_elapsed:.2f} Seconds",
        f"{reward_mean:.2f}",
        f"{plus_minus_std_deviation:.2f}",
        str(convergence_count),
        f"{success_count}/{settings['num_episodes']}",
        f"{best:.2f}",
        f"{worst:.2f}",
    )
    table.add_row(
        f"Recent {settings['log_interval']}",
        f"{time_elapsed:.2f} Seconds",
        f"{r_reward_mean:.2f}",
        f"{r_plus_minus_std_deviation:.2f}",
        str(convergence_count),
        f"{r_success_count}/{settings['log_interval']}",
        f"{r_best:.2f}",
        f"{r_worst:.2f}",
    )
    console.print(table)



def plot_rewards(episode_rewards, window=100):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode reward')
    rolling_mean = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    plt.plot(np.arange(window - 1, len(episode_rewards)), rolling_mean,
             label=f'Rolling average (window size = {window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()