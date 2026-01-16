import gymnasium as gym
from stable_baselines3 import DQN
from number_guesser_env import number_guesser_env

env = number_guesser_env()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
