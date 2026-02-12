import gymnasium as gym
import simpy
import random
import numpy as np

# Incorporating multiple orders with different sizes, start times, and due dates, 
# and different routes through the FUs. As only 2 FUs and sequential order, routes can be: A only, B only, A then B.

MIN_ORDER_SIZE = 3
MAX_ORDER_SIZE = 6
LATEST_ORDER_START = 20
LATEST_ORDER_DUE = 100
NUM_ORDERS = 2 
FU = {"A": {"index": 1, "service_time": 4},
    "B": {"index": 2, "service_time": 5},
    "C": {"index": 3, "service_time": 2},
    "D": {"index": 4, "service_time": 3},
    "E": {"index": 5, "service_time": 3},
    "F": {"index": 6, "service_time": 2},
}
 # FU A has id 1 and processing time 3, FU B has id 2 and processing time 6
times = [FU[machine]["service_time"] for machine in FU] # extract processing times for each FU
max_time = max(times) # maximum processing time among FUs, used for calculating remaining times in the state representation

class WorkshopEnv(gym.Env):
    def __init__(self, max_steps=100, fu_config=None, custom_orders=None):
        super().__init__()
        self.max_steps = max_steps  
        self.fu_config = fu_config if fu_config is not None else FU
        self.custom_orders = custom_orders  # will be a dict or None   
        # Use len(self.FU_config) and maybe len(custom_orders) for spaces
        self.FU_names = list(self.fu_config.keys())
        num_fus = len(self.FU_names)
        num_orders = NUM_ORDERS if custom_orders is None else len(custom_orders)   
        # Define action and observation space
        # Example: 2 actions for each FU, 0: hold, 1: request to start
        self.action_space = gym.spaces.MultiDiscrete([num_orders+2]*num_fus, dtype=np.int32)
        obs_high = []
        obs_low = []
        obs_shape = 0
        # Observation space: [Norm_to_do, Norm_complete, Norm_ttd, 
        for i in range(num_orders):
            obs_high.extend([1,1,1])
            obs_low.extend([0,0,-1])
            obs_high.extend([1] * len(FU))   # binary: which FU starts this order's route
            obs_low.extend([0] * len(FU))
            obs_shape += 3 + len(FU)
        # busy_A, Norm_waiting_time_overall, Norm_working_time_A_overall
        for i in range(len(FU)):
            obs_high.extend([1,1,1])
            obs_low.extend([0,0,0])
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
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(len(self.fu_config))] 
        self.job_queue = [] # track pending jobs
        self.reward = 0.0

        self.orders = {} # track orders as a dictionary with order_id as key
        self.FUs = {}
        if self.custom_orders is not None:
            # assume dict: {order_id: {"size":..., "start_time":..., "due_date":..., "route":[...]} }
            self.orders = self.custom_orders.copy()
            num_orders = len(self.orders)
        else:
            for i in range(NUM_ORDERS):
                size = np.random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE + 1)
                start_time = i * np.random.randint(0, LATEST_ORDER_START + 1)
                due_date = np.random.randint(start_time + max_time*(size+1), LATEST_ORDER_DUE + 1)

                self.orders[i+1] = {
                    "size": size,
                    "start_time": start_time,
                    "due_date": due_date,
                    "route": [],
                    "to_do": size,
                    "complete": 0
                    }

                while len(self.orders[i+1]["route"]) == 0:
                    route = []
                    choice = list(FU.keys())
                    for j in range(len(FU)): # for each FU
                        selection = random.choice([choice[j], None]) # randomly choose whether to include the FU in the route for the order
                        if selection is not None:
                            route.append(selection)
                    self.orders[i+1]["route"] = route # add route
        for i in range(len(self.fu_config)):
            name = self.FU_names[i]
            self.FUs[name] = {
                    "service_time": FU[name]["service_time"],
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
        self.working = 0
        self.time_log = []
        self.fu_log = {name: {"idx": [], "util_rate": []} for name in self.FU_names}
        self.order_log = {order: {"to_do": [], "complete": []} for order in self.orders}

        # track position of each component in each order in the system, 0: not started, 1: in A, 2: B, 3: completed
        self.position = {}
        self.step_count = 0
        for i in range(NUM_ORDERS):
            order_size = self.orders[i+1]["size"]
            self.position[i+1] = np.zeros((order_size, 3), dtype=np.int32) # initialize position for each part in the order
            for j in range(order_size): # for each part in the order
                self.position[i+1][j][0] = j + 1 # part number in order
                self.position[i+1][j][1] = 0 # initial position
                self.position[i+1][j][2] = 0 # initial processing time
        return self.summary(), {}

    def summary(self):
        obs = []
        total_time = max(self.env.now, 1e-6)  # avoid divide-by-zero
        for i in range(NUM_ORDERS):
            size = self.orders[i+1]["size"]
            time_to_deadline = self.orders[i+1]["due_date"] - self.env.now # calculate time to deadline for each order
            order_length = self.orders[i+1]["due_date"] - self.orders[i+1]["start_time"]
            if order_length == 0:
                norm_ttd = 0
            else:
                norm_ttd = time_to_deadline / order_length # normalise time to deadline
            norm_ttd = np.clip(norm_ttd, -1, 1)
            obs.append(self.orders[i+1]["to_do"]/size) # normalise orders to do as a fraction of the order size
            obs.append(self.orders[i+1]["complete"]/size)
            obs.append(norm_ttd)
            first_fu = self.orders[i+1]["route"][0]
            for fu_name in self.FU_names:
                obs.append(1.0 if fu_name == first_fu else 0.0)

       
        for fu_name in self.FU_names:
            fu = self.FUs[fu_name]
            busy_util = fu["working_time"] / total_time
            # if you want explicit wait/system utilisations, sum their lists:
            wait_sum = sum(fu["waiting_times"]) if fu["waiting_times"] else 0.0
            wait_util = wait_sum / total_time
            # clip to [0,1] to respect Box bounds
            busy_util = float(np.clip(busy_util, 0.0, 1.0))
            wait_util = float(np.clip(wait_util, 0.0, 1.0))
            obs = np.append(
                obs,
                [fu["busy"], busy_util, wait_util]
                )
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
                order_id, part_idx = self.FUs[fu_name]["working_on"]
                if order_id != 0:
                    label = f"O{order_id}-P{part_idx}"
                else:
                    label = ""
                self.fu_log[fu_name]["idx"].append(label)
                self.fu_log[fu_name]["util_rate"].append(self.FUs[fu_name]["util_rate"])
            for order in self.orders:
                self.order_log[order]["to_do"].append(self.orders[order]["to_do"])
                self.order_log[order]["complete"].append(self.orders[order]["complete"])

    # If env time jumps backwards (shouldn't happen), we do nothing

    
    def job(self, env, machine_id, order_id):
        machine_idx = FU[machine_id]["index"] - 1
        # gets the arrival time for that item going into a specific FU
        self.FUs[machine_id]["arrival_times"].append(self.env.now)
        with self.machines[machine_idx].request() as req:
            #request the machine to start
            yield req
            if machine_id == self.orders[order_id]["route"][0]: # if first FU in route of that order
                self.FUs[machine_id]["busy"] = 1
                self.working += 1
                self.orders[order_id]["to_do"] -= 1 # change to only A start
                part_idx = self.orders[order_id]["size"] - self.orders[order_id]["to_do"]
                #self.position[order_id][part_idx-1][1] = machine_idx + 1 # add in updates to position
                self.FUs[machine_id]["working_on"] = [order_id,part_idx]  

            else:                       
                self.FUs[machine_id]["busy"] = 1
                previous_machine_id = self.FUs[machine_id]["waiting"][0] # get previous machine in route
                self.FUs[previous_machine_id]["busy"] = 0  # free up previous machine when moving to next
                self.FUs[machine_id]["waiting"].pop(0) # remove from wait list only if not first in route
                self.FUs[machine_id]["working_on"] = self.FUs[previous_machine_id]["working_on"] 
                self.FUs[previous_machine_id]["working_on"] =[0,0]

            self.reward += 0.5 # reward for starting a job
            self.FUs[machine_id]["start_service_times"].append(self.env.now)
            service_time = self.FUs[machine_id]["service_time"]
            yield env.timeout(service_time)
            self.FUs[machine_id]["working_time"] += service_time
            self.FUs[machine_id]["departure_times"].append(self.env.now)
            self.FUs[machine_id]["util_rate"] = self.FUs[machine_id]["working_time"] / self.env.now
            self.reward += 1 # reward for completing a job at any FU, in future can differentiate between FUs
            if machine_id == self.orders[order_id]["route"][-1]: # if it's the last FU then reward for completion 
                self.reward += 5
                self.orders[order_id]["complete"] += 1
                self.FUs[machine_id]["busy"] = 0
                self.working -= 1
                self.FUs[machine_id]["working_on"]=[0,0]
            else:
                next_machine_id = self.orders[order_id]["route"][self.orders[order_id]["route"].index(machine_id) + 1] # get next machine in route
                self.FUs[next_machine_id]["waiting"].append(machine_id)

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
        for i in reversed(range(len(action))):
            name = self.FU_names[i]
            if action[i] == 0: # hold 
                # if self.FUs[i]["remaining_times"] <= 0 and to_do_total > 0:
                #     self.reward -= 0.5 # penalty for holding when there are jobs to do and machine is free
                if self.FUs[name]["busy"] == 1 and self.FUs[name]["remaining_time"] > 0:
                    self.reward +=2 # machine correctly working on a job 
            elif action[i] == 1: # request to start FU i from previous FU
                # request to start FU i
                if self.FUs[name]["busy"] == 0 and len(self.FUs[name]["waiting"]) > 0: # if FU is free and previous FU in route is complete
                    previous_fu = self.FUs[name]["waiting"][0]   
                    order_id = self.FUs[previous_fu]["working_on"][0]
                    self.FUs[name]["remaining_time"] = self.FUs[name]["service_time"]
                    self.env.process(self.job(self.env, name, order_id))
                else:
                    self.reward -= 1 # penalty for requesting to start A when it's busy or there are no jobs to do
            elif action[i] > 1: # request to start FU i from order with id action[i]
                order_id = action[i] - 1
                # FU must be free, there must be orders to do,  and this must be the first FU of the order route
                if self.FUs[name]["busy"] == 0 and \
                    self.orders[order_id]["to_do"] > 0 and \
                    self.orders[order_id]["route"][0] == name and \
                    self.orders[order_id]["start_time"]<=self.env.now: 
                    self.FUs[name]["remaining_time"] = self.FUs[name]["service_time"]
                    self.env.process(self.job(self.env, name, order_id))
                    self.reward += 1 # reward for starting a job
                else:
                    self.reward -= 0.1 # penalty for requesting to start when invalid
 
        prev_time = int(self.env.now)

        active_times = [self.FUs[fu_name]["remaining_time"] 
                        for fu_name in self.FUs 
                        if self.FUs[fu_name]["remaining_time"] > 0]
        if len(active_times) > 0:
            service_time = min(active_times)
            self.env.run(until=self.env.now + service_time)  # Run until a machine is complete it's job
            for fu_name in self.FUs:
                self.FUs[fu_name]["remaining_time"] -= service_time  # may produce negative times but that just indicates how long it's been idle
        elif self.working > 0: # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
            self.env.run(until=self.env.now + 1) # if no machines are working just run for a time step
        else:
            total_remaining = sum(o["to_do"] for o in self.orders.values())
            if total_remaining > 0:
                self.env.run(until=self.env.now + 1)  # Advance time so agent can take next action
        self._log_gantt(prev_time, int(self.env.now))
        #print(self.order_log)


        lateness_penalty = 0.0
        for oid, order in self.orders.items():
            # only penalise orders that have arrived
            if self.env.now > order["start_time"]:
                lateness = max(0.0, self.env.now - order["due_date"])
                lateness_penalty += 0.05 * lateness 
            else:
                earliness = max(0.0, order["due_date"] - self.env.now)
                lateness_penalty -= 0.05 * earliness

        self.reward -= lateness_penalty
        total_orders = 0
        complete_orders = 0
        for i in self.orders:
            total_orders += self.orders[i]["size"]
            complete_orders += self.orders[i]["complete"]
        if total_orders == complete_orders:
            terminated = True
            avg_util = 0
            for fu_name in self.FU_names:
                avg_util += self.FUs[fu_name]["util_rate"]
            avg_util /= len(self.FU_names)
            self.reward += 300*avg_util
            #print("All orders completed!")
            # calc utilisation rate and take away based on percentage

        return self.summary(), self.reward, terminated, truncated, info