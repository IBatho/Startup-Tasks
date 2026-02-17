from stable_baselines3 import PPO
from Task22 import WorkshopEnv

env = WorkshopEnv()  # same obs/action spaces as before
model = PPO.load("ppo_factory_policy_continued.zip", env=env)

# Continue training
model.learn(total_timesteps=50000)  # adds 50k more steps on top
model.save("ppo_factory_policy_continued2.zip")
