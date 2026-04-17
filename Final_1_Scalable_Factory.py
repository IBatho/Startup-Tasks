import gymnasium as gym
import simpy
import random
import numpy as np
import pandas as pd
from collections import namedtuple

# Hard Limits for the running of operations as a proof of concept
# These limits can be adjusted 
# They are designed to limit the range of possibilities that the agents needs to be trained on for quicker training
MIN_ORDER_SIZE = 3
MAX_ORDER_SIZE = 15
LATEST_ORDER_START = 30
LATEST_ORDER_DUE = 400
MAX_NUM_ORDERS = 4 
MAX_NUM_FUS = 10
MAX_JOB_TIME = 30

RouteStep = namedtuple("RouteStep", ["fu_name", "service_time"])
PartID = namedtuple("PartID", ["order_id", "part_index"])

class WorkshopEnv(gym.Env):
    def __init__(self, max_steps=800, fu_config=None, custom_orders=None):
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
        self.action_space = gym.spaces.MultiDiscrete([MAX_NUM_ORDERS+2]*MAX_NUM_FUS, dtype=np.int32)
        obs_high = []
        obs_low = []
        obs_shape = 0
        # Observation space: [Norm_to_do, Norm_complete, Norm_ttd, 
        for i in range(MAX_NUM_ORDERS):
            obs_high.extend([1,1,1])
            obs_low.extend([-1,-1,-1])
            obs_high.extend([1] * MAX_NUM_FUS)   # binary: which FU starts this order's route
            obs_low.extend([0] * MAX_NUM_FUS)
            obs_shape += 3 + MAX_NUM_FUS
        # busy_A, Norm_waiting_time_overall, Norm_working_time_A_overall
        for i in range(MAX_NUM_FUS):
            obs_high.extend([1,1,1])
            obs_low.extend([0,0,-1])
            obs_shape += 3
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
                "waiting": [], # identifies which FUs want to send an item 
                "working_on": [0,0],
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
            for fu_name in self.FU_names:
                obs.append(1.0 if fu_name == first_fu else 0.0)
            for _ in range(self.FU_spare):
                obs.append(0.0) # add in dummy values for missing FUs for each order
        for _ in range(self.Order_spare):
            obs.extend([0,0,0]) # add in dummy values for missing orders
            obs.extend([0] * MAX_NUM_FUS) # add in dummy values for missing orders for each FU

        for fu_name in self.FU_names:
            fu = self.fu_config[fu_name]
            busy_util = fu["working_time"] / total_time
            # if you want explicit wait/system utilisations, sum their lists:
            order = fu["working_on"][0] # get order currently being worked on
            if order != 0:
                idx = next((i for i, step in enumerate(self.orders[order]["route"]) if step.fu_name == fu_name), None)    
                order_time = self.orders[order]["route"][idx].service_time if idx is not None else 0
            else: 
                order_time = 0
            remaining_norm = fu["remaining_time"] / order_time if order_time > 0 else 0
            # clip to [0,1] to respect Box bounds
            busy_util = float(np.clip(busy_util, 0.0, 1.0))
            remaining_norm = float(np.clip(remaining_norm, -1.0, 1.0))
            obs.extend([fu["busy"], busy_util, remaining_norm])
        for _ in range(self.FU_spare):
            obs.extend([0,0,0]) # add in dummy values for missing FUs
        
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
                order_id, part_idx = self.fu_config[fu_name]["working_on"]
                if order_id != 0:
                    label = f"O{order_id}-P{part_idx}"
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
            if machine_id == self.orders[order_id]["route"][0].fu_name: # if first FU in route of that order
                self.fu_config[machine_id]["busy"] = 1
                self.working += 1
                self.orders[order_id]["to_do"] -= 1 # change to only A start
                part_idx = self.orders[order_id]["size"] - self.orders[order_id]["to_do"]
                #self.position[order_id][part_idx-1][1] = machine_idx + 1 # add in updates to position
                self.fu_config[machine_id]["working_on"] = [order_id,part_idx]  

            else:                       
                self.fu_config[machine_id]["busy"] = 1
                previous_machine_id = self.fu_config[machine_id]["waiting"][0] # get previous machine in route
                self.fu_config[previous_machine_id]["busy"] = 0  # free up previous machine when moving to next
                self.fu_config[machine_id]["waiting"].pop(0) # remove from wait list only if not first in route
                self.fu_config[machine_id]["working_on"] = self.fu_config[previous_machine_id]["working_on"] 
                self.fu_config[previous_machine_id]["working_on"] =[0,0]

            self.reward += 0.5 # reward for starting a job
            self.fu_config[machine_id]["start_service_times"].append(self.env.now)
            service_time = self.service_time(machine_id, order_id)
            yield env.timeout(service_time)
            self.fu_config[machine_id]["working_time"] += service_time
            self.fu_config[machine_id]["departure_times"].append(self.env.now)
            self.fu_config[machine_id]["util_rate"] = self.fu_config[machine_id]["working_time"] / self.env.now
            self.reward += 1 # reward for completing a job at any FU, in future can differentiate between FUs
            if machine_id == self.orders[order_id]["route"][-1].fu_name: # if it's the last FU then reward for completion 
                self.reward += 10
                self.orders[order_id]["complete"] += 1
                self.fu_config[machine_id]["busy"] = 0
                self.working -= 1
                self.fu_config[machine_id]["working_on"]=[0,0]
            else:
                current_machined_idx = next(i for i, step in enumerate(self.orders[order_id]["route"]) if step.fu_name == machine_id)
                next_machine_id = self.orders[order_id]["route"][current_machined_idx + 1].fu_name # get next machine in route
                self.fu_config[next_machine_id]["waiting"].append(machine_id)

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
        to_do_total = sum(self.orders[o]["to_do"] for o in self.orders)

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
                # if self.fu_config[name]["remaining_time"] <= 0 and to_do_total > 0:
                #     self.reward -= 0.05 # penalty for holding when there are jobs to do and machine is free
                if self.fu_config[name]["busy"] == 1 and self.fu_config[name]["remaining_time"] > 0:
                    self.reward += 1 # machine correctly working on a job 
            elif action_i == 1: # request to start FU i from previous FU
                # request to start FU i
                if self.fu_config[name]["busy"] == 0 and len(self.fu_config[name]["waiting"]) > 0: # if FU is free and previous FU in route is complete
                    previous_fu = self.fu_config[name]["waiting"][0]   
                    order_id = self.fu_config[previous_fu]["working_on"][0]
                    self.fu_config[name]["remaining_time"] = self.service_time(name, order_id)
                    self.env.process(self.job(self.env, name, order_id))
                else:
                    self.reward -= 1 # penalty for requesting to start A when it's busy or there are no jobs to do
            elif action_i > 1: # request to start FU i from order with id action[i]
                if action_i - 1 <= self.num_orders:
                    order_id = action_i - 1
                    # FU must be free, there must be orders to do,  and this must be the first FU of the order route
                    if self.fu_config[name]["busy"] == 0 and \
                        self.orders[order_id]["to_do"] > 0 and \
                        self.orders[order_id]["route"][0].fu_name == name and \
                        self.orders[order_id]["start_time"]<=self.env.now: 
                        self.fu_config[name]["remaining_time"] = self.service_time(name, order_id)
                        self.env.process(self.job(self.env, name, order_id))
                        self.reward += 1 # reward for starting a job
                    else:
                        self.reward -= 0.05 # penalty for requesting to start when invalid
    
        prev_time = int(self.env.now)

        active_times = [self.fu_config[fu_name]["remaining_time"] 
                        for fu_name in self.fu_config 
                        if self.fu_config[fu_name]["remaining_time"] > 0]

        total_busy = sum([self.fu_config[name]["busy"] for name in self.FU_names])
        service_time = 0.0
        if total_busy == self.num_fus and len(active_times):  # if all machines are busy, run until the next one is free
            service_time = min(active_times)
            #self.env.run(until=self.env.now + 1)
            self.env.run(until=self.env.now + service_time +0.001)  # Run until a machine is complete it's job
        elif self.working > 0: # if machines are working 
            self.env.run(until=self.env.now + 0.25) # if no machines are working just run for a time step
            service_time = 0.25
        else:
            total_remaining = sum(o["to_do"] for o in self.orders.values())
            if total_remaining > 0:
                self.env.run(until=self.env.now + 1) # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
                #self.env.run(until=self.env.now + 1)  # Advance time so agent can take next action
        for fu_name in self.fu_config:
            if self.fu_config[fu_name]["busy"] == 1:    
                self.fu_config[fu_name]["remaining_time"] -= service_time  # may produce negative times but that just indicates how long it's been idle
        
        self._log_gantt(prev_time, int(self.env.now))
        #print(self.order_log)            

        total_orders = 0
        complete_orders = 0
        for id, order in self.orders.items():
            total_orders += order["size"]
            complete_orders += order["complete"]
            lateness_penalty = 0.0

            if order["complete"] == order["size"] and not order["complete_true"]:
                order["complete_true"] = True
                lateness = self.env.now - order["due_date"]
                lateness_penalty += 0.1 * lateness
                print(f"Order {id} completed at time {self.env.now:.2f} with lateness {lateness:.2f} and penalty {lateness_penalty:.2f} at {self.step_count} for {info}")
                self.reward -= lateness_penalty # penalty for lateness, can adjust multiplier to make more or less important
        if total_orders == complete_orders:
            terminated = True
            avg_util = 0
            for fu_name in self.FU_names:
                avg_util += self.fu_config[fu_name]["util_rate"]
            avg_util /= len(self.FU_names)
            self.reward += 100*avg_util
            print("All orders completed!")
            # calc utilisation rate and take away based on percentage
        #print(f"Step {self.step_count}: Time {self.env.now:.2f}, Reward: {self.reward:.2f}, Lateness Penalty: {lateness_penalty:.2f}, Total Orders: {total_orders}, Complete Orders: {complete_orders}")
        return self.summary(), self.reward, terminated, truncated, info