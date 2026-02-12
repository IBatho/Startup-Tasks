import gymnasium as gym
from stable_baselines3 import DQN, PPO
import time
import numpy as np
import matplotlib.pyplot as plt
from Task22 import WorkshopEnv
import csv
from stable_baselines3.common.callbacks import BaseCallback

env = WorkshopEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[256, 256]),  # Larger network
    tensorboard_log="./ppo_factory_tb",
)

class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.steps = []
        self.ep_rew_mean = []
        self.losses = []
        self._printed_keys = False


    def _on_step(self) -> bool:
        # training logs live in self.logger.name_to_value
        logs = self.logger.name_to_value
        t = self.num_timesteps
        # Temporary: print keys once to see what's available
        if not self._printed_keys and len(logs) > 0:
            print("LOG KEYS:", list(logs.keys()))
            self._printed_keys = True

        if "train/loss" in logs:
            self.steps.append(t)
            self.losses.append(logs["train/loss"])

        # Log reward when available (only some steps)
        if "rollout/ep_rew_mean" in logs:
            self.ep_rew_mean.append((t, logs["rollout/ep_rew_mean"]))
        
        for k in ["rollout/ep_rew_mean", "train/episode_reward"]:
            if k in logs:
                self.ep_rew_mean.append((t, logs[k]))
                break

        return True

log_cb = LogCallback()
model.learn(total_timesteps=10000, callback=log_cb)
model.save("ppo_factory_policy.zip")
#env.save("ppo_factory_env.pkl")  # optional: only if you want to reload with same env wrapper


num_eval_episodes = 30
episode_indices = []
episode_returns = []

for ep in range(num_eval_episodes):
    obs, info = env.reset()
    done = False
    ep_ret = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        done = terminated or truncated

    episode_indices.append(ep)
    episode_returns.append(ep_ret)

# Plot reward curve
plt.figure()
plt.plot(episode_indices, episode_returns, marker="o")
plt.xlabel("Evaluation episode")
plt.ylabel("Total episode reward")
plt.title("PPO: evaluation episode returns")
plt.grid(True)
plt.show()

# Loss curve
plt.figure()
plt.plot(log_cb.steps, log_cb.losses)
plt.xlabel("Timesteps")
plt.ylabel("Loss")
plt.title("PPO training: loss")
plt.grid(True)
plt.show()

# obs, info = env.reset()
# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("t:", env.env.now, "A working_on:", env.FUs["A"]["working_on"])
#     if terminated or truncated:
#         break

# print(env.time_log[:10])
# print({k: v[:10] for k, v in env.fu_log.items()})

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    #env.render()
    print(obs, reward, terminated, truncated, info)
    if terminated or truncated:
        #obs, info = env.reset()
        if terminated:
            reason = "terminated"
        else:
            reason = "truncated"
        print(f"ended due to {reason} at time {env.env.now}")
        break#


time_log = env.time_log
fu_log = env.fu_log  # dict: fu_name -> list of labels
order_log = env.order_log


# Ensure all FU lists are same length as time_log
max_len = len(time_log)
for fu_name in fu_log:
    if len(fu_log[fu_name]["idx"]) < max_len:
        fu_log[fu_name]["idx"] += [""] * (max_len - len(fu_log[fu_name]["idx"]))
        fu_log[fu_name]["util_rate"] += [0.0] * (max_len - len(fu_log[fu_name]["util_rate"]))

# Ensure all order lists are same length as time_log
for order in order_log:
    if len(order_log[order]["to_do"]) < max_len:
        order_log[order]["to_do"] += [0] * (max_len - len(order_log[order]["to_do"]))
        order_log[order]["complete"] += [0] * (max_len - len(order_log[order]["complete"]))



header = ["time"] 
for fu_name in env.FU_names:
    header += f"FU {fu_name}", "Utilisation Rate"
for order in env.orders:
    header += f"to do_{order}", f"complete_{order}"

rows = []
for idx, t in enumerate(time_log):
    row = [t]
    for fu_name in env.FU_names:
        row.append(fu_log[fu_name]["idx"][idx])
        row.append(fu_log[fu_name]["util_rate"][idx])
    for order in env.orders.keys():
        row.append(order_log[order]["to_do"][idx])
        row.append(order_log[order]["complete"][idx])
    rows.append(row)
results = []


with open("gantt_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)