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
    success_rate = f"{num_episodes_passed / episode_span * 100:.2f%}"

    best_reward = f"{np.max(episode_rewards[-episode_span:]):.2f%}"
    worst_reward = f"{np.min(episode_rewards[-episode_span:])::.2f%}"

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