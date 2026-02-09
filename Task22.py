import gymnasium as gym
import simpy
import random
import numpy as np

# Incorporating multiple orders with different sizes, start times, and due dates, 
# and different routes through the FUs. As only 2 FUs and sequential order, routes can be: A only, B only, A then B.

MIN_ORDER_SIZE = 1
MAX_ORDER_SIZE = 5
LATEST_ORDER_START = 20
LATEST_ORDER_DUE = 100
NUM_ORDERS = 2 
FU = {"A":{"index":1, "service_time":3}, "B":{"index":2, "service_time":6}} # FU A has id 1 and processing time 3, FU B has id 2 and processing time 6
times = [FU[machine]["service_time"] for machine in FU] # extract processing times for each FU
max_time = max(times) # maximum processing time among FUs, used for calculating remaining times in the state representation

class WorkshopEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.max_steps = max_steps        
        # Define action and observation space
        # Example: 2 actions for each FU, 0: hold, 1: request to start
        self.action_space = gym.spaces.Box(
            low = -1,
            high = NUM_ORDERS,
            shape=(len(FU),),
            dtype=np.int32
        )
        # Observation space: [to_do_O1, complete_O1, Time_to_deadline_01, to_do_O2, complete_O2, Time_to_deadline_02, busy_A, waiting_time_A, system_time_A, working_time_A, busy_B, waiting_time_B, system_time_B, working_time_B]   
        self.observation_space = gym.spaces.Box(
            low=np.zeros(14, dtype=np.float32), 
            high=np.array([MAX_ORDER_SIZE,MAX_ORDER_SIZE,LATEST_ORDER_DUE,
                           MAX_ORDER_SIZE,MAX_ORDER_SIZE,LATEST_ORDER_DUE,
                           1,50,1000,max_time*MAX_ORDER_SIZE*2,
                           1,50,1000,max_time*MAX_ORDER_SIZE*2], 
                           dtype=np.float32), 
            shape=(14,), 
            dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(len(FU))] 
        self.job_queue = [] # track pending jobs

        self.orders = {} # track orders as a dictionary with order_id as key
        self.FUs = {}
        for i in range(NUM_ORDERS):
            size = np.random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE + 1)
            start_time = i * np.random.randint(0, LATEST_ORDER_START + 1)
            due_date = np.random.randint(start_time + max_time*(size+1), LATEST_ORDER_DUE + 1)

            self.orders[i+1] = {
                "size": size,
                "start_time": start_time,
                "due_date": due_date,
                "route": [],
                "to do": size,
                "complete": 0
                }

            accept = 0
            while accept == 0:
                route = []
                choice = list(FU.keys())
                for j in range(len(FU)): # for each FU
                    selection = random.choice([choice[j], None]) # randomly choose whether to include the FU in the route for the order
                    if selection is not None:
                        route.append(selection)
                    else: accept += 1
            self.orders[i+1]["route"] = route # add route
        self.FU_names = list(FU.keys())
        for i in range(len(FU)):
            name = self.FU_names[i]
            self.FUs[name] = {
                    "service_time": FU[name]["service_time"],
                    "busy": 0, # with which order, 0: nothing, 1: order 1
                    "arrival_times": [],
                    "start_service_times": [],
                    "departure_times": [],
                    "remaining_times": [],
                    "waiting_times": [],
                    "system_times": [],
                    "working_time": 0,
                    "waiting": [] # identifies which FUs want to send an item 
                    }
        self.working = 0
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
        avg_waiting_times = []
        avg_system_times = []
        total_working_times = []
        obs = []
        no_jobs = 0
        for fu_name, fu_data in self.FUs.items():
            n_arrivals = len(fu_data["arrival_times"])
            n_started = len(fu_data["start_service_times"])
            n_finished = len(fu_data["departure_times"])
            no_jobs += n_arrivals
            if n_finished > 0:
                avg_waiting_times.append(np.mean([s - a for a, s in zip(fu_data["arrival_times"], fu_data["start_service_times"])]))
                avg_system_times.append(np.mean([d - a for a, d in zip(fu_data["arrival_times"], fu_data["departure_times"])]))
                total_working_times.append(sum([d - a for a, d in zip(fu_data["start_service_times"], fu_data["departure_times"])]))
            else:
                avg_waiting_times.append(0.0)
                avg_system_times.append(0.0)
                total_working_times.append(0.0)
        for i in range(NUM_ORDERS):
            time_to_deadline = self.orders[i+1]["due_date"] - self.env.now # calculate time to deadline for each order
            obs.append(self.to_do[i+1])
            obs.append(self.orders[i+1]["complete"])
            obs.append(time_to_deadline)
        for i, fu_name in enumerate(self.FU_names):
            obs = np.append(obs, [self.FUs[fu_name]["busy"], avg_waiting_times[i], avg_system_times[i], total_working_times[i]])
        obs.flatten()
        return obs
    
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
                self.orders[order_id]["to do"] -= 1 # change to only A start
            else:                       #
                self.FUs[machine_id]["busy"] = 1
                previous_machine_id = self.orders[order_id]["route"][self.orders[order_id]["route"].index(machine_id) - 1] # get previous machine in route
                self.FUs[previous_machine_id]["busy"] = 0  # free up previous machine when moving to next
            self.FUs[machine_id]["waiting"].pop(0) # remove from wait list
            self.reward += 0.5 # reward for starting a job
            self.FUs[machine_id]["start_service_times"].append(self.env.now)
            service_time = self.FUs[machine_id]["service_time"]
            yield env.timeout(service_time)
            self.FUs[machine_id]["departure_times"].append(self.env.now)
            self.reward += 1 # reward for completing a job at any FU, in future can differentiate between FUs
            if machine_id == self.orders[order_id]["route"][-1]: # if it's the last FU then reward for completion 
                self.reward += 5
                self.orders[order_id]["complete"] += 1
                self.FUs[machine_id]["busy"] = 0
                self.working -= 1
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
                "working": self.working}
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
        for i in range(len(action)):
            name = self.FU_names[i]
            if action[i] == -1: # hold 
                # if self.FUs[i]["remaining_times"] <= 0 and to_do_total > 0:
                #     self.reward -= 0.5 # penalty for holding when there are jobs to do and machine is free
                if self.FUs[name]["busy"] == 1 and self.FUs[name]["remaining_times"] > 0:
                    self.reward +=1 # machine correctly working on a job 
            elif action[i] == 0: # request to start FU i from previous FU
                # request to start FU i
                if self.FUs[name]["busy"] == 0 and self.FUs[name]["waiting"] > 0: # if FU is free and previous FU in route is complete
                    self.FUs[name]["remaining_times"] = self.FUs[name]["service_time"]
                    self.env.process(self.job(self.env, self.FUs[name], self.FUs[name]["waiting"][0]))
                else:
                    self.reward -= 0.5 # penalty for requesting to start A when it's busy or there are no jobs to do
            elif action[i] > 0: # request to start FU i from order with id action[i]
                order_id = action[i]
                # FU must be free, their maust be orders to do,  and this must be the first FU of the order route
                if self.FUs[name]["busy"] == 0 and self.orders[order_id ]["to do"] > 0 and self.orders[order_id]["route"][0] == i: 
                    self.env.process(self.job(self.env, f"Job_{i}_{self.step_count}", self.FUs[i]))
                    self.reward += 0.5 # reward for starting a job
                else:
                    self.reward -= 0.5 # penalty for requesting to start when invalid
 
        active_times = [self.FUs[fu_name]["remaining_times"] 
                        for fu_name in self.FUs 
                        if self.FUs[fu_name]["remaining_times"] > 0]
        if len(active_times) > 0:
            service_time = min(active_times)
            self.env.run(until=self.env.now + service_time)  # Run until a machine is complete it's job
            for fu_name in self.FUs:
                self.FUs[fu_name]["remaining_times"] -= service_time  # may produce negative times but that just indicates how long it's been idle
        elif self.working == 1: # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
            self.env.run(until=self.env.now + 1) # if no machines are working just run for a time step
        terminated = True
        for fu_name in self.FUs:
            if self.FUs[fu_name]["to do"] == self.FUs[fu_name]["complete"]:
                self.reward += 200
                print("All orders completed!")
            else:
                terminated = False

        return self.summary(), self.reward, terminated, truncated, info