import gymnasium as gym
from stable_baselines3 import DQN, PPO
import time
import numpy as np
import matplotlib.pyplot as plt
from Task22 import WorkshopEnv as Case1
import csv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
from Case_study_1 import WorkshopEnv as Case2
from Final_1_Scalable_Factory import WorkshopEnv as Case3


case_studies = {
    "1": Case1,
    "2": Case2,
    "3": Case3
}


print("Which case study do you want to run?")
case_study = input("1 - Fixed Factory, \n 2 for Scalable Factory - small model \n 3 for Scalable Factory - larger model\n")

if case_study in case_studies:
    env_class = case_studies[case_study]
else:
    print("Invalid case study selection.")
    exit()

training_start_point = input("Are you starting or continuing training? (type 'start' or 'continue') ")
if training_start_point == "start":
    model = None
else:
    model_file_name = input("Enter the model file name and folder path to load (e.g., './logs/Case_1_ppo_factory_policy_1500000_steps.zip'): ")

save_path = input("Enter the folder path to save the trained model (e.g., './logs/'): ")

time_steps = input("Enter the number of training timesteps (e.g., 300000): ")


def my_env():
    env = env_class()
    return Monitor(env)

env = DummyVecEnv([my_env])
eval_cb = EvalCallback(env, best_model_save_path="./logs/",
                       log_path="./logs/", eval_freq=1000,
                       deterministic=True, render=False)

if training_start_point == "start":
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0002,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network
        tensorboard_log=f"./ppo_tensorboard/{case_study}_ppo_tensorboard/".format(case_study=case_study),
    )   # build a fresh model
elif training_start_point == "continue":
    if os.path.exists(model_file_name):
        model = PPO.load(model_file_name, env=env)
    else:
        print(f"File '{model_file_name}' not found.")
        exit()
else:
    print(f"Invalid choice '{training_start_point}'.")
    exit()

class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.steps = []
        self.losses = []
        self.rew_steps = []
        self.ep_rew_mean = []
        self._printed_keys = False

    def _on_step(self) -> bool:
        logs = self.logger.name_to_value
        t = self.num_timesteps

        if not self._printed_keys and len(logs) > 0:
            print("LOG KEYS:", list(logs.keys()))
            self._printed_keys = True

        if "train/loss" in logs:
            self.steps.append(t)
            self.losses.append(logs["train/loss"])

        if "rollout/ep_rew_mean" in logs:
            self.rew_steps.append(t)
            self.ep_rew_mean.append(logs["rollout/ep_rew_mean"])

        return True

log_cb = LogCallback()
start = time.time()
model.learn(total_timesteps=int(time_steps), callback=log_cb, reset_num_timesteps=False)
model.save(save_path)
print(f"Training time: {time.time() - start:.2f}s")
#env.save("ppo_factory_env.pkl")  # optional: only if you want to reload with same env wrapper


# num_eval_episodes = 1000
# episode_indices = []
# episode_returns = []

# for ep in range(num_eval_episodes):
#     obs, info = env.reset()
#     done = False
#     ep_ret = 0.0

#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#         ep_ret += reward
#         done = terminated or truncated

#     episode_indices.append(ep)
#     episode_returns.append(ep_ret)

# Plot reward curve
plt.figure()
plt.plot(log_cb.rew_steps, log_cb.ep_rew_mean, marker="o")
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

obs = env.reset()
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


# time_log = env.time_log
# fu_log = env.fu_log  # dict: fu_name -> list of labels
# order_log = env.order_log


# # Ensure all FU lists are same length as time_log
# max_len = len(time_log)
# for fu_name in fu_log:
#     if len(fu_log[fu_name]["idx"]) < max_len:
#         fu_log[fu_name]["idx"] += [""] * (max_len - len(fu_log[fu_name]["idx"]))
#         fu_log[fu_name]["util_rate"] += [0.0] * (max_len - len(fu_log[fu_name]["util_rate"]))

# # Ensure all order lists are same length as time_log
# for order in order_log:
#     if len(order_log[order]["to_do"]) < max_len:
#         order_log[order]["to_do"] += [0] * (max_len - len(order_log[order]["to_do"]))
#         order_log[order]["complete"] += [0] * (max_len - len(order_log[order]["complete"]))



# header = ["time"] 
# for fu_name in env.FU_names:
#     header += f"FU {fu_name}", "Utilisation Rate"
# for order in env.orders:
#     header += f"to do_{order}", f"complete_{order}"

# rows = []
# for idx, t in enumerate(time_log):
#     row = [t]
#     for fu_name in env.FU_names:
#         row.append(fu_log[fu_name]["idx"][idx])
#         row.append(fu_log[fu_name]["util_rate"][idx])
#     for order in env.orders.keys():
#         row.append(order_log[order]["to_do"][idx])
#         row.append(order_log[order]["complete"][idx])
#     rows.append(row)
# results = []


# with open("gantt_log.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerows(rows)