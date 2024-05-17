import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Initialize the vectorized environment with multiple parallel instances
vec_env = make_vec_env("CartPole-v1", n_envs=8)  # Set n_envs to 16 or as per your resource limits

# Define the model with specified hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=32,                # The number of steps to run for each environment per update
    batch_size=256,               # Minibatch size
    gae_lambda=0.8,             # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    gamma=0.98,                 # Discount factor
    n_epochs=20,                  # Number of epochs when optimizing the surrogate loss
    ent_coef=0.0,                # Entropy coefficient for exploration vs exploitation trade-off
    learning_rate=0.001,
    clip_range= 0.2,
)

# Training
should_train = input("Do you want to train the model? (y/n): ")

if should_train.lower() == "y":
    model.learn(total_timesteps=int(1e5))  # 100,000

model.save("ppo_cartpole_v1")

# Clearing memory
del model

# Load model
model = PPO.load("ppo_cartpole_v1")

# Reset environment
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(("Rewards: ", rewards))
    vec_env.render("human")