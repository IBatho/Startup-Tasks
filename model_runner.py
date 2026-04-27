from Final_1_Scalable_Factory import WorkshopEnv as Case3
from Case_study_1 import WorkshopEnv as Case2
from Task22 import WorkshopEnv as Case1

case_studies = {
    "1": Case1,
    "2": Case2,
    "3": Case3
}

# from Task22 import WorkshopEnv
from stable_baselines3 import DQN, PPO
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import namedtuple
RouteStep = namedtuple("RouteStep", ["fu_name", "service_time"])

print("Which case study do you want to run?")
case_study = input("1 - Fixed Factory, \n 2 for Scalable Factory - small model \n 3 for Scalable Factory - larger model\n")

if case_study in case_studies:
    env_class = case_studies[case_study]
else:
    print("Invalid case study selection.")
    exit()

model_file_name = input("Enter the model file name and folder path to load (e.g., './logs/Case_1_ppo_factory_policy_1500000_steps.zip'): ")


def FU(index, default_service_time):
    return {
        "index": index,
        "default_service_time": default_service_time,
        "busy": 0, # with which order, 0: nothing, 1: order 1
        "arrival_times": [],
        "start_service_times": [],
        "departure_times": [],
        "remaining_time": 0.0,
        "waiting_times": [],
        "system_times": [],
        "working_time": 0.0,
        "waiting": [], # identifies which FUs want to send an item 
        "working_on": [0,0],
        "util_rate": 0.0
    }

def parse_route(route_str):
    steps = []
    for step in route_str.split(">"):
        fu, time = step.split(":")
        steps.append(RouteStep(fu.strip(), float(time)))
    return steps

orders = "./orders/Case{case_study}_orders.csv".format(case_study=case_study)
my_orders = {}

if case_study != "1":
    FUs = "./FUs/Case{case_study}_FUs.csv".format(case_study=case_study)
    my_FU = {}
    with open(FUs, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            fu_name = row[0]
            index = int(row[1])
            default_service_time = float(row[2])
            my_FU[fu_name] = FU(index, default_service_time)

    with open(orders, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            order_id = int(row[0])
            size = int(row[1])
            start_time = int(row[2])
            due_date = int(row[3])
            route = row[4]
            to_do = int(row[5])
            complete = int(row[6])
            complete_true = row[7].lower() == "true"
            my_orders[order_id] = {
                "size": size,
                "start_time": start_time,
                "due_date": due_date,
                "route": parse_route(route),
                "to_do": to_do,
                "complete": complete,
                "complete_true": complete_true
            }

else:
    my_FU = None
    with open(orders, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            order_id = int(row[0])
            size = int(row[1])
            start_time = int(row[2])
            due_date = int(row[3])
            route = row[4]
            to_do = int(row[5])
            complete = int(row[6])
            my_orders[order_id] = {
                "size": size,
                "start_time": start_time,
                "due_date": due_date,
                "route": [fu.strip() for fu in route.split(">")],
                "to_do": to_do,
                "complete": complete,
            }


env = case_studies[case_study](fu_config=my_FU, custom_orders=my_orders)
loaded_model = PPO.load(model_file_name, env=env)

obs, info = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs, reward, terminated, truncated, info)


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
# Define specific colors for each order
color_map = {
    1: 'green',  # Or 'tab:green' for a slightly softer shade
    2: 'blue',    # Or 'tab:blue' for a slightly softer shade
    3: 'orange'  # Or 'tab:orange' for a slightly softer shade
}

fig, ax = plt.subplots(figsize=(20, 4)) # Increased width for better spacing

graph_FUs = env.FU_names

for i, fu_name in enumerate(graph_FUs):
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
                            fontweight='bold', fontsize=16)
            
            start_time = t
            current_job = status



# Formatting
ax.set_yticks(range(len(graph_FUs)))
ax.set_yticklabels([f"FU {n}" for n in graph_FUs], fontsize=30, fontweight="bold")
ax.set_xlabel("Time →", fontsize=30, fontweight="bold")
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20) # Optional: ensures y-ticks match
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.grid(True, axis='x', linestyle=':', alpha=0.5)

# Legend to explain the colors

legend_patches = [mpatches.Patch(color=color_map[oid], label=f'Order {oid}') for oid in unique_orders]
leg = ax.legend(handles=legend_patches, 
          loc='center right',           
          title="Orders",
          title_fontsize=20,           # Makes the "Orders" title bigger
          fontsize=18,                 # Makes the "Order 1" & "Order 2" text bigger
          handlelength=2.5,            # Makes the color rectangle wider
          handleheight=1.5,            # Makes the color rectangle taller
          labelspacing=1.2,            # Adds more vertical space between the orders
          borderpad=1.2,
          frameon=True,                
          edgecolor='black',           
          facecolor='white',           
          framealpha=1.0,              
          shadow=True)

# Set the border thickness separately
leg.get_frame().set_linewidth(1.5)
plt.tight_layout()
plt.show()

with open("gantt_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
