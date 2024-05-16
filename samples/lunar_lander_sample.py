import time
import gym

env = gym.make('LunarLander-v2', render_mode='human')
print("Welcome to the Lunar Lander simulation!")


def print_readable_state(state):
    print(f"Position (x, y): ({state[0]:.2f}, {state[1]:.2f}), Velocity (x, y): ({state[2]:.2f}, {state[3]:.2f}),")
    print(f"Angle: {state[4]:.2f}, Angular Velocity: {state[5]:.2f}, Legs Contact: (left: {bool(state[6])}, right: {bool(state[7])})")


for i in range(40):
    state = env.reset()
    done = False
    total_reward = 0
    start_time = time.time()  # Start the timer before the loop begins
    actions_taken = 0
    while not done:
        action = 0
        state, reward, done, _, _ = env.step(action)
        print_readable_state(state)
        actions_taken += 1
        total_reward += reward
        elapsed_time = time.time() - start_time  # Calculate the elapsed time since the loop started
        print(f"reward: {reward} / {total_reward} Elapsed time: {actions_taken}/{elapsed_time:.2f} seconds")  # Print the elapsed time with two decimal places
    print("DEAD")

