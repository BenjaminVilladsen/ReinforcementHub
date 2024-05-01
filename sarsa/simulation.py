import gymnasium as gym
from config_and_helpers import discretize, epsilon_greedy_policy, n_bins, state_bounds, Q

def run_simulation(episodes=100, render_mode="human"):
    env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
                   turbulence_power=1.5, render_mode=render_mode)
    for i in range(episodes):
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        total_reward = 0

        while True:
            action = epsilon_greedy_policy(state, Q, epsilon=0)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = discretize(observation, n_bins, state_bounds)

            if done or truncated:
                print(f"Simulation episode {i + 1}: Reward: {total_reward}")
                break

if __name__ == "__main__":
    run_simulation()
