from Task22 import WorkshopEnv
from stable_baselines3 import DQN, PPO
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
loaded_model = PPO.load("ppo_factory_policy_L0.0003_T400000.zip", env=env)

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
rows = []

'''for fu_name in env.FU_names:
    header += f"FU {fu_name}", "Utilisation Rate"
for order in env.orders:
    header += f"to do_{order}", f"complete_{order}"

for idx, t in enumerate(time_log):
    row = [t]
    for fu_name in env.FU_names:
        row.append(fu_log[fu_name]["idx"][idx])
        row.append(fu_log[fu_name]["util_rate"][idx])
    for order in env.orders.keys():
        row.append(order_log[order]["to_do"][idx])
        row.append(order_log[order]["complete"][idx])
    rows.append(row)
results = []'''


# ... (keep your existing simulation code above) ...

# 1. Prepare Transposed Data for CSV
# The first row will be the Header: "FU/Time", followed by all time steps
header = ["FU / Time"] + [str(t) for t in time_log]
rows = []
unique_orders = list(my_orders.keys())
colors = plt.cm.get_cmap('tab10', len(unique_orders))
color_map = {order_id: colors(i) for i, order_id in enumerate(unique_orders)}

fig, ax = plt.subplots(figsize=(16, 8)) # Increased width for better spacing

for i, fu_name in enumerate(env.FU_names):
    status_list = fu_log[fu_name]["idx"]
    
    # Track continuous blocks to avoid drawing many tiny 1-unit rectangles
    # and to center the text label once per operation
    current_job = None
    start_time = 0
    
    for t, status in enumerate(status_list + [""]): # Added empty string to flush last block
        if status != current_job:
            if current_job: # End of a block
                duration = t - start_time
                order_id_str = ''.join(filter(str.isdigit, current_job.split('-')[0]))
                order_id = int(order_id_str) if order_id_str else None
                job_color = color_map.get(order_id, "gray")
                
                # Draw the bar
                ax.broken_barh([(start_time, duration)], (i - 0.35, 0.7), 
                               facecolors=job_color, edgecolor='black', linewidth=0.5)
                
                # 2. Readable Labels: Only show the Order ID (e.g., "O1")
                # and only if the block is wide enough to fit it
                if duration > 1:
                    ax.text(start_time + duration/2, i, f"{current_job}", 
                            ha='center', va='center', color='white', 
                            fontweight='bold', fontsize=9)
            
            start_time = t
            current_job = status

# Formatting
ax.set_yticks(range(len(env.FU_names)))
ax.set_yticklabels([f"FU {n}" for n in env.FU_names])
ax.set_xlabel("Time →")
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

# 3. Add a Legend to explain the colors
legend_patches = [mpatches.Patch(color=color_map[oid], label=f'Order {oid}') for oid in unique_orders]
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Orders")

plt.tight_layout()
plt.show()

with open("gantt_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
