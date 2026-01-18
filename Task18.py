import gymnasium as gym
from stable_baselines3 import DQN, PPO
from number_guesser_env import number_guesser_env

env = number_guesser_env(render_mode="human")
# model = DQN("MlpPolicy", 
#             env, 
#             verbose=1, 
#             exploration_fraction=0.2,
#             exploration_final_eps=0.05,)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(
        net_arch=[256, 256]  # Larger network
    )
)
model.learn(total_timesteps=200000)
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
