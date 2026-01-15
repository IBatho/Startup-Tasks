import gymnasium as gym
from grid_world_env import GridWorldEnv
from stable_baselines3 import DQN

env = GridWorldEnv(render_mode="human")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

obs, info = env.reset()
env.render()

obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()


'''for _ in range(10):
    action = env.action_space.sample() # randomly choses a value corrasponding to an action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
        env.render()'''
#env.close()


    

