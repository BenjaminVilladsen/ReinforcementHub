#pip install swig
#pip install gymnasium[all]

import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   #action = env.action_space.sample()  # this is where you would insert your policy
   action = 2
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()