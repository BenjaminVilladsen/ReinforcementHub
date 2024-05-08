import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Initialize the vectorized environment with multiple parallel instances
vec_env = make_vec_env("MountainCar-v0", n_envs=16)  # Set n_envs based on your hardware capabilities

# Define the PPO model with specified hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,  # Pass the vectorized environment directly to the model
    learning_rate=0.001,  # Adjust this learning rate according to your requirements
    n_steps=2048,  # Number of steps to run for each environment per update
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Training
should_train = input("Do you want to train the model? (y/n): ")
if should_train.lower() == "y":
    model.learn(total_timesteps=int(1e6))  # 1 million timesteps

model.save("ppo_mountain_car_v0")

# Clear memory
del model

# Load the model
model = PPO.load("ppo_mountain_car_v0", env=vec_env)

# Reset environment and run a simple interaction loop
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(f"Rewards: {rewards}")
    vec_env.render("human")
