import gymnasium as gym
import simpy
import random
import numpy as np

MIN_ORDER_SIZE = 1
MAX_ORDER_SIZE = 5
LATEST_ORDER_START = 20
LATEST_ORDER_DUE = 100
NUM_ORDERS = 2 
FU = {"A":[1, 3], "B":[2, 6]} # FU A has id 1 and processing time 3, FU B has id 2 and processing time 6
times = [machine[1] for machine in FU.values()] # extract processing times for each FU
max_time = max(times) # maximum processing time among FUs, used for calculating remaining times in the state representation

class WorkshopEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.max_steps = max_steps        
        # Define action and observation space
        # Example: 4 actions, 0: hold, 
        # 1: request to start A, 
        # 2: request to start B from A, 
        # 3: request to start both A and B
        self.action_space = gym.spaces.Discrete(4) 
        # Observation space: [to_do_O1, to_do_O2, Time_to_deadline_01, Time_to_deadline_01, busy_A, waiting_time_A, system_time_A, working_time_A, busy_B, waiting_time_B, system_time_B, working_time_B]   
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float32), 
            high=np.array([MAX_ORDER_SIZE,MAX_ORDER_SIZE,LATEST_ORDER_DUE,LATEST_ORDER_DUE,1,50,1000,max_time*MAX_ORDER_SIZE*2,1,50,1000,max_time*MAX_ORDER_SIZE*2], dtype=np.float32), 
            shape=(12,), 
            dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(2)] # in future can use different capacities from a list of capacities for each FU]
        self.job_queue = [] # track pending jobs

        self.orders = [[],[]] # track 
        self.to_do = np.array([0, 0], dtype=np.int32) # number of orders for each FU
        for i in range(NUM_ORDERS):
            self.orders[i].append(i + 1) # add order id
            self.orders[i].append(np.random.randint(MIN_ORDER_SIZE, MAX_ORDER_SIZE + 1)) # add number in order
            self.to_do[i] += self.orders[i][1] # update total number of orders
            self.orders[i].append(i * np.random.randint(0, LATEST_ORDER_START + 1)) # add start date
            self.orders[i].append(np.random.randint(self.orders[i][2] + max_time*(self.orders[i][1]+1), LATEST_ORDER_DUE + 1)) # add due date
        self.orders = np.array(self.orders, dtype=np.float32)
        self.complete = {1:0, 2:0} # number of completed orders for each order
        self.busy = {"A":0, "B":0} # busy status of each FU
        self.arrival_times = {"A":[], "B":[]} # track arrival times for each FU
        self.start_service_times = {"A":[], "B":[]} # track start service times for each FU
        self.departure_times = {"A":[], "B":[]} # track departure times for each FU
        self.remaining_times = {"A":0.0, "B":0.0} # track remaining processing times for each FU
        self.working = 0
        self.position = np.empty((NUM_ORDERS, 4), dtype=np.int16) # track position of each order in the system, 0: not started, 1: in A, 2: B, 3: completed
        for i in range(NUM_ORDERS):
            self.position[i][0] = self.orders[i][0] # order id
            self.position[i][1] = i + 1 # part number in order
            self.position[i][2] = 0 # initial position
            self.position[i][3] = 0 # initial processing time


        return self.summary(), {}

    def summary(self):
        avg_waiting_times = []
        avg_system_times = []
        total_working_times = []
        obs = []
        for i in range(len(self.orders)):
            time_to_deadline = [d - self.env.now for d in self.orders[i][3]] # calculate time to deadline for each order
            obs.append(self.to_do[i])
            obs.append(self.complete[i])
            obs.append(time_to_deadline[i])
        for i in range(len(FU)):
            obs = np.append(obs, [self.busy[i], avg_waiting_times[i], avg_system_times[i], total_working_times[i]])
        obs.flatten()
        return obs
    
    def job(self, env, name, machine_id):
        with self.machines[machine_id].request() as req:
            #request the machine to start
            yield req
            #start processing

    def step(self, action):
        self.reward = 0