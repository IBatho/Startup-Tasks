import gymnasium as gym
from stable_baselines3 import DQN
from number_guesser_env import number_guesser_env

env = number_guesser_env(render_mode="human")
model = DQN("MlpPolicy", 
            env, 
            verbose=1, 
            exploration_fraction=0.2,
            exploration_final_eps=0.05,)
model.learn(total_timesteps=100000)
obs, info = env.reset()

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(obs, reward, terminated, truncated, info)
    if terminated or truncated:
        obs, info = env.reset()
        print(terminated, truncated)


'''for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    if done or truncated:
        print("episode ended")
        break'''
