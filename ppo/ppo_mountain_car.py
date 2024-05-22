import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Initialize the vectorized environment with multiple parallel instances
vec_env = make_vec_env("MountainCar-v0", n_envs=16)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# Define the model with specified hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=256,              # Increased from 16
    gae_lambda=0.98,
    normalize_advantage=True,
    gamma=0.99,
    n_epochs=4,
    ent_coef=0.0,
    learning_rate=0.0003      # Added learning rate
)

# Training
should_train = input("Do you want to train the model? (y/n): ")

if should_train.lower() == "y":
    model.learn(total_timesteps=int(2e6))  # Increased to 2 million
    model.save("ppo_mountain_car_v0")
    print("Model training completed and saved.")

# Clear memory
del model

# Load model
model = PPO.load("ppo_mountain_car_v0")

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
        obs, rewards, dones, infos = vec_env.step(action)
        episode_rewards += rewards
        # Note: Rendering a vectorized environment might not be supported in the "human" mode
        vec_env.render("human")
        # Alternatively, you can render one of the environments if needed:
        #vec_env.envs[0].render("human")
        done = dones
    total_rewards.append(sum(episode_rewards))

# Print evaluation results
average_reward = sum(total_rewards) / num_episodes
print(f"Average Reward over {num_episodes} episodes: {average_reward}")

# Close the environment
vec_env.close()
