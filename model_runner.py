from Task22 import WorkshopEnv
from stable_baselines3 import DQN, PPO
import csv


my_FU = {
    "A": {"index": 1, "service_time": 4},
    "B": {"index": 2, "service_time": 5},
    "C": {"index": 3, "service_time": 2},
    "D": {"index": 4, "service_time": 3},
    "E": {"index": 5, "service_time": 3},
    "F": {"index": 6, "service_time": 2},
}

my_orders = {
    1: {"size": 6, "start_time": 0, "due_date": 75, "route": ["A", "C", "D", "E"], "to_do": 6, "complete": 0},
    2: {"size": 5, "start_time": 5, "due_date": 90, "route": ["B", "C", "D", "F"], "to_do": 5, "complete": 0},
}

env = WorkshopEnv(fu_config=my_FU, custom_orders=my_orders)
loaded_model = PPO.load("ppo_factory_policy.zip", env=env)

obs, info = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

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
