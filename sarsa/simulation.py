import gymnasium as gym
from config_and_helpers import discretize, epsilon_greedy_policy, n_bins, state_bounds
from playsound import playsound


############################################################
#                FOR RUNNING THE SIMULATION                #
############################################################

def run_simulation(policy_Q):
    env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0,
                   turbulence_power=1.5, render_mode='human')
    for i in range(100):
        observation, info = env.reset()
        state = discretize(observation, n_bins, state_bounds)
        total_reward = 0

        while True:
            action = epsilon_greedy_policy(state, policy_Q, epsilon=0, env=env)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = discretize(observation, n_bins, state_bounds)

            if done or truncated:
                print(f"Simulation episode {i + 1}: Reward: {total_reward}")
                if total_reward > 200:
                    playsound('../sarsa_new/sound.mp3')
                break


if __name__ == "__main__":
    run_simulation()
