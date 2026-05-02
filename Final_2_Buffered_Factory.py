import gymnasium as gym
import simpy
import random
import numpy as np
import pandas as pd
from collections import namedtuple

# Incorporating multiple orders with different sizes, start times, and due dates, 
# and different routes through the FUs. As only 2 FUs and sequential order, routes can be: A only, B only, A then B.

MIN_ORDER_SIZE = 3
MAX_ORDER_SIZE = 6
LATEST_ORDER_START = 20
LATEST_ORDER_DUE = 500
MAX_NUM_ORDERS = 4 
MAX_NUM_FUS = 10
MAX_JOB_TIME = 5
MAX_JOBS = MAX_NUM_FUS*MAX_ORDER_SIZE
MAX_BUFFER_SIZE = 5

RouteStep = namedtuple("RouteStep", ["fu_name", "service_time"])
PartID = namedtuple("PartID", ["order_id", "part_index"])

class WorkshopEnv(gym.Env):
    def __init__(self, max_steps=400, fu_config=None, custom_orders=None):
        super().__init__()
        self.max_steps = max_steps
        self.fu_config = fu_config  # will be a dict or None
        if self.fu_config is None:
            self.custom = False
        else:
            self.custom = True
        self.custom_orders = custom_orders  # will be a dict or None   
        # Define action and observation space
        # Example: 2 actions for each FU, 0: hold, 1: request to start
        self.action_space = gym.spaces.MultiDiscrete([MAX_NUM_ORDERS+1]*MAX_NUM_FUS, dtype=np.int32)
        obs_high = []
        obs_low = []
        obs_shape = 0
        # Observation space: [Norm_to_do, Norm_complete, Norm_ttd, Norm_FU_id]
        for i in range(MAX_NUM_ORDERS):
            obs_high.extend([1,1,1,1])
            obs_low.extend([-1,-1,-1,0])
            obs_shape += 4
        # busy_A, Norm_waiting_time_overall, Norm_working_time_A_overall
        for i in range(MAX_NUM_FUS):
            obs_high.extend([1,1,1])
            obs_low.extend([0,0,0])
            obs_high.extend([1] * MAX_BUFFER_SIZE)   # track which jobs are waiting for this FU, up to MAX_BUFFER_SIZE
            obs_low.extend([0] * MAX_BUFFER_SIZE)
            obs_shape += 3 + MAX_BUFFER_SIZE
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32), 
            high=np.array(obs_high, dtype=np.float32),
            shape=(obs_shape,),
            dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()
        self.num_fus = random.randint(1, MAX_NUM_FUS) if not self.custom else len(self.fu_config)
        self.FU_spare = MAX_NUM_FUS - self.num_fus
        if self.custom is False:
            self.fu_config = {f"FU_{i+1}": {
                "index": i+1, 
                "default_service_time": random.randint(1, 5),
                "busy": 0, # with which order, 0: nothing, 1: order 1
                "arrival_times": [],
                "start_service_times": [],
                "departure_times": [],
                "remaining_time": 0.0,
                "waiting_times": [],
                "system_times": [],
                "working_time": 0.0,
                "waiting": [PartID(0,0)] * MAX_BUFFER_SIZE, # identifies the parts waiting for this FU, up to MAX_BUFFER_SIZE, with dummy values for empty slots
                "working_on": PartID(0,0), # identifies the part currently being processed by this FU, with dummy value if idle
                "util_rate": 0.0
                } for i in range(self.num_fus)}
        self.FU_names = list(self.fu_config.keys())
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(self.num_fus)] 
        self.job_queue = [] # track pending jobs
        self.reward = 0.0

        self.orders = {} # track orders as a dictionary with order_id as key
        if self.custom_orders is not None:
            # assume dict: {order_id: {"size":..., "start_time":..., "due_date":..., "route":[...]} }
            self.orders = self.custom_orders.copy()
            self.num_orders = len(self.orders)
        else:
            self.num_orders = random.randint(1, MAX_NUM_ORDERS)
            for i in range(self.num_orders):
                size = np.random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE + 1)
                start_time = i * np.random.randint(0, LATEST_ORDER_START + 1)
                route = []
                highest_service_time = 0
                longest_FU = 0
                while len(route) == 0: # ensure at least one FU in route
                    for j in self.FU_names: # for each FU
                        selection = random.choice([j, None]) # randomly choose whether to include the FU in the route for the order
                        if selection is not None:
                            time = random.uniform(1, MAX_JOB_TIME)
                            highest_service_time = max(highest_service_time, time)
                            if highest_service_time == time:
                                longest_FU = selection
                            route.append(RouteStep(selection, time))
                times = sorted([step.service_time for step in route])
                other_times = times[:-1]
                min_time = highest_service_time * size + sum(other_times) # minimum time to process all orders in an order
                low = start_time + min_time # earliest due date is after the minimum time to process all items in the order, can be adjusted with a multiplier if you want to make it easier or harder
                due_date = np.random.randint(low, max(low+1, LATEST_ORDER_DUE + 1))
    
                self.orders[i+1] = {
                    "size": size,
                    "start_time": start_time,
                    "due_date": due_date,
                    "route": route,
                    "to_do": size,
                    "complete": 0,
                    "complete_true": False
                    }
        self.Order_spare = MAX_NUM_ORDERS - self.num_orders
        self.working = 0
        self.time_log = []
        self.fu_log = {name: {"idx": [], "util_rate": []} for name in self.FU_names}
        self.order_log = {order: {"to_do": [], "complete": []} for order in self.orders}

        # track position of each component in each order in the system, 0: not started, 1: in A, 2: B, 3: completed
        self.parts = {}
        for i in self.orders.keys():
            order_size = self.orders[i]["size"]
            for j in range(order_size): # for each part in the order
                self.parts[PartID(i, j+1)] = {"position": 0, "processing_time": 0} # part number in order
        self.step_count = 0
        return self.summary(), {}

    def summary(self):
        obs = []
        total_time = max(self.env.now, 1e-6)  # avoid divide-by-zero
        for i in self.orders:
            size = self.orders[i]["size"]
            time_to_deadline = self.orders[i]["due_date"] - self.env.now # calculate time to deadline for each order
            order_length = self.orders[i]["due_date"] - self.orders[i]["start_time"]
            if order_length == 0:
                norm_ttd = 0
            else:
                norm_ttd = time_to_deadline / order_length # normalise time to deadline
            norm_ttd = np.clip(norm_ttd, -1, 1)
            obs.append(self.orders[i]["to_do"]/size) # normalise orders to do as a fraction of the order size
            obs.append(self.orders[i]["complete"]/size)
            obs.append(norm_ttd)
            first_fu = self.orders[i]["route"][0].fu_name
            first_fu_idx = self.fu_config[first_fu]["index"] / MAX_NUM_FUS # normalise FU index to [0,1]
            obs.append(first_fu_idx)


        for _ in range(self.Order_spare):
            obs.extend([0,0,0,0]) # add in dummy values for missing orders

        for fu_name in self.FU_names:
            fu = self.fu_config[fu_name]
            busy_util = fu["working_time"] / total_time
            # if you want explicit wait/system utilisations, sum their lists:
            wait_sum = sum(fu["waiting_times"]) if fu["waiting_times"] else 0.0
            wait_util = wait_sum / total_time
            # clip to [0,1] to respect Box bounds
            busy_util = float(np.clip(busy_util, 0.0, 1.0))
            wait_util = float(np.clip(wait_util, 0.0, 1.0))
            obs.extend([fu["busy"], busy_util, wait_util])
            for i in range(MAX_BUFFER_SIZE):
                order_id = fu["waiting"][i].order_id
                job_idx = order_id/MAX_NUM_ORDERS if order_id != 0 else 0.0
                obs.append(job_idx)
        for _ in range(self.FU_spare):
            obs.extend([0,0,0]) # add in dummy values for missing FUs
            for i in range(MAX_BUFFER_SIZE):
                obs.append(0) # add in dummy values for missing FUs
        
        return np.array(obs, dtype=np.float32).flatten()
    
    def _log_gantt(self, start_t: int, end_t: int):
        """
        For each integer second t in [start_t, end_t),
        record what each FU is working on.
        Label format: 'O{order_id}-P{part_index}' or '' if idle.
        """
        # ensure logs are continuous in time
        for t in range(len(self.time_log), end_t):
            self.time_log.append(t)
            for fu_name in self.FU_names:
                # if FU is busy, use working_on to build label; else blank
                part_id = self.fu_config[fu_name]["working_on"]
                if part_id != PartID(0,0):
                    label = f"O{part_id.order_id}-P{part_id.part_index}"
                else:
                    label = ""
                self.fu_log[fu_name]["idx"].append(label)
                self.fu_log[fu_name]["util_rate"].append(self.fu_config[fu_name]["util_rate"])
            for order in self.orders:
                self.order_log[order]["to_do"].append(self.orders[order]["to_do"])
                self.order_log[order]["complete"].append(self.orders[order]["complete"])

    # If env time jumps backwards (shouldn't happen), we do nothing
    def service_time(self, fu_name, order_id):
        time = next((step.service_time for step in self.orders[order_id]["route"] if step.fu_name == fu_name), self.fu_config[fu_name]["default_service_time"])
        return time
    
    def job(self, env, machine_id, order_id):
        machine_idx = self.fu_config[machine_id]["index"] - 1
        # gets the arrival time for that item going into a specific FU
        self.fu_config[machine_id]["arrival_times"].append(self.env.now)
        with self.machines[machine_idx].request() as req:
            #request the machine to start
            yield req
            self.working += 1
            if machine_id == self.orders[order_id]["route"][0].fu_name: # if first FU in route of that order
                self.orders[order_id]["to_do"] -= 1 # change to only A start
                part_idx = self.orders[order_id]["size"] - self.orders[order_id]["to_do"]
                #self.position[order_id][part_idx-1][1] = machine_idx + 1 # add in updates to position
                self.fu_config[machine_id]["working_on"] = PartID(order_id, part_idx)
            else:                       
                part_idx = self.fu_config[machine_id]["working_on"].part_index
            service_time = self.service_time(machine_id, order_id)
            self.fu_config[machine_id]["busy"] = 1
            self.fu_config[machine_id]["remaining_time"] = service_time


            self.reward += 0.5 # reward for starting a job
            self.fu_config[machine_id]["start_service_times"].append(self.env.now)
            yield env.timeout(service_time)
            self.working -= 1
            self.fu_config[machine_id]["working_on"] = PartID(0,0)
            self.fu_config[machine_id]["working_time"] += service_time
            self.fu_config[machine_id]["departure_times"].append(self.env.now)
            self.fu_config[machine_id]["util_rate"] = self.fu_config[machine_id]["working_time"] / self.env.now
            self.reward += 1 # reward for completing a job at any FU, in future can differentiate between FUs
            
            order_idx = next(i for i, o in enumerate(self.orders[order_id]["route"]) if o.fu_name == machine_id) # find where in the route this FU is for the order
            next_FU_in_route = self.orders[order_id]["route"][order_idx + 1].fu_name if order_idx < len(self.orders[order_id]["route"]) - 1 else None
            if next_FU_in_route is None: # if it's the last FU then reward for completion 
                self.reward += 10
                self.orders[order_id]["complete"] += 1
                self.fu_config[machine_id]["busy"] = 0
                self.fu_config[machine_id]["working_on"]=PartID(0,0)
            else: # if there is space in the buffer for the next FU
                while PartID(0,0) not in self.fu_config[next_FU_in_route]["waiting"]: # wait until there is space in the buffer  
                    yield env.timeout(0.1) # check every 0.1 time units
                idx = next(i for i, step in enumerate(self.fu_config[next_FU_in_route]["waiting"]) if step == (0,0)) # find the first empty slot in the buffer
                self.fu_config[next_FU_in_route]["waiting"][idx] = PartID(order_id, part_idx) # add the part to the buffer of the next FU
                self.fu_config[machine_id]["busy"] = 0
                self.fu_config[machine_id]["working_on"]=PartID(0,0)
                self.parts[PartID(order_id, part_idx)]["position"] = next_FU_in_route + "_buffer" # update position of part in system
                
                
                
    
    def step(self, action):
        # let simpy run the environment and define rewards
        terminated = False
        truncated = False
        info = {"total_orders": {k: v["size"] for k, v in self.orders.items()}, 
                "time": self.env.now, 
                "action": action,
                "working": self.working,
                "start date": {k: v["start_time"] for k, v in self.orders.items()},
                "due date": {k: v["due_date"] for k, v in self.orders.items()}
                }
        self.step_count += 1
        self.reward = 0
        #to_do_total = sum(self.to_do)

        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1
            return self.summary(), reward, terminated, truncated, info
        # in each action set up a job
        # action is a list of values indicating which FUs to hold or request to start
        # -1 would hold
        # 0 would move from previous FU to that FU if possible (e.g. move from A to B)
        # any higher numbers would request to start that FU from the order with that number's id
        
        for i in reversed(range(self.num_fus)):
            name = self.FU_names[i]
            action_i = int(action[i])
            if action_i == 0: # hold 
                # if self.FUs[i]["remaining_times"] <= 0 and to_do_total > 0:
                #     self.reward -= 0.5 # penalty for holding when there are jobs to do and machine is free
                if self.fu_config[name]["busy"] == 1 and self.fu_config[name]["remaining_time"] > 0:
                    self.reward += 1 # machine correctly working on a job 
                else:
                    self.reward -= 0.5 # small penalty for holding when not working on a job
            elif action_i <= self.num_orders:
                order_id = action_i
                # FU must be free 
                if self.fu_config[name]["busy"] == 0:
                    # chekc if it's first FU in order or from buffer there must be orders to do,  and this must be the first FU of the order route
                    if self.orders[order_id]["route"][0].fu_name == name and \
                        self.orders[order_id]["start_time"]<=self.env.now and \
                        self.orders[order_id]["to_do"] > 0:
                        self.env.process(self.job(self.env, name, order_id))
                        self.reward += 1 # reward for starting a job
                    elif order_id in [step.order_id for step in self.fu_config[name]["waiting"]]: # check if order_id called is in the buffer
                        idx = next(i for i, step in enumerate(self.fu_config[name]["waiting"]) if step.order_id == order_id) # find where in the buffer the job is waiting
                        part_id = self.fu_config[name]["waiting"][idx] # find the part number waiting for that order in the buffer
                        self.parts[part_id]["position"] = name # update position of part in system
                        self.parts[part_id]["processing_time"] = self.service_time(name, order_id)
                        self.fu_config[name]["waiting"][idx] = PartID(0,0) # replace in buffer with (0,0)
                        self.fu_config[name]["working_on"] = part_id # update working on to the part that was waiting
                        self.env.process(self.job(self.env, name, order_id))
                        self.reward += 1 # reward for starting a job
                else:
                    self.reward -= 0.1 # penalty for requesting to start when invalid

        prev_time = int(self.env.now)

        active_times = [self.fu_config[fu_name]["remaining_time"] 
                        for fu_name in self.fu_config 
                        if self.fu_config[fu_name]["remaining_time"] > 0]

        total_busy = sum([self.fu_config[name]["busy"] for name in self.FU_names])

        if total_busy == self.num_fus and len(active_times):  # if all machines are busy, run until the next one is free
            service_time = min(active_times)
            #self.env.run(until=self.env.now + 1)
            self.env.run(until=self.env.now + service_time +0.001)  # Run until a machine is complete it's job
            for fu_name in self.fu_config:
                self.fu_config[fu_name]["remaining_time"] -= service_time  # may produce negative times but that just indicates how long it's been idle
        elif self.working > 0: # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
            self.env.run(until=self.env.now + 0.25) # if no machines are working just run for a time step
        else:
            total_remaining = sum(o["to_do"] for o in self.orders.values())
            if total_remaining > 0:
                self.env.run(until=self.env.now + 1) # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
                #self.env.run(until=self.env.now + 1)  # Advance time so agent can take next action
        self._log_gantt(prev_time, int(self.env.now))
        #print(self.order_log)            

        
        total_orders = 0
        complete_orders = 0
        for id, order in self.orders.items():
            total_orders += order["size"]
            complete_orders += order["complete"]
            lateness_penalty = 0.0

            if order["complete"] == order["size"] and order["complete_true"] == False:
                order["complete_true"] = True
                lateness = self.env.now - order["due_date"]
                lateness_penalty += 0.5 * lateness
                self.reward -= lateness_penalty
        if total_orders == complete_orders:
            terminated = True
            avg_util = 0
            for fu_name in self.FU_names:
                avg_util += self.fu_config[fu_name]["util_rate"]
            avg_util /= len(self.FU_names)
            self.reward += 300*avg_util
            #print("All orders completed!")
            # calc utilisation rate and take away based on percentage
        print(f"Step {self.step_count}: Time {self.env.now:.2f}, Reward: {self.reward:.2f}, Lateness Penalty: {lateness_penalty:.2f}, Total Orders: {total_orders}, Complete Orders: {complete_orders}")
        return self.summary(), self.reward, terminated, truncated, info