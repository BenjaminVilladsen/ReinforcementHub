import matplotlib.pyplot as plt
import numpy as np
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