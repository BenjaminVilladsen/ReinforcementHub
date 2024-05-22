import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Initialize the vectorized environment with multiple parallel instances
vec_env = make_vec_env("LunarLander-v2", n_envs=16)

# Define the model with specified hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01
)

# Training
should_train = input("Do you want to train the model? (y/n): ")

if should_train.lower() == "y":
    model.learn(total_timesteps=int(1e6))  # 1 million
    model.save("ppo_lunar_lander_v2")
    print("Model training completed and saved.")

# Clear memory
del model

# Load model
model = PPO.load("ppo_lunar_lander_v2")

# Reset environment
obs = vec_env.reset()

# Evaluation loop
total_rewards = []
num_episodes = 10  # Evaluate over 10 episodes

for episode in range(num_episodes):
    obs = vec_env.reset()
    episode_rewards = 0
    done = [False] * vec_env.num_envs
    while not all(done):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        episode_rewards += rewards
        vec_env.render("human")
    total_rewards.append(episode_rewards)

# Print evaluation results
average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")

# Close the environment
vec_env.close()
