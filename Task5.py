import gymnasium as gym
from grid_world_env import GridWorldEnv

env = GridWorldEnv()
obs, info = env.reset()

for _ in range(10):
    action = env.action_space.sample() # randomly choses a value corrasponding to an action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    env.render()
env.close()
    

