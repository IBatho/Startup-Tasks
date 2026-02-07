import gymnasium as gym
import simpy
import random
import numpy as np

MIN_ORDERS = 1
MAX_ORDERS = 15
LATEST_ORDER_START = 20
LATEST_ORDER_DUE = 100
FU = {"A":[0, 3], "B":[1, 5]} # FU A has id 0 and processing time 3, FU B has id 1 and processing time 5
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
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float32), 
            high=np.array([MAX_ORDERS,MAX_ORDERS,1,50,50,1000,1,50,50,1000], dtype=np.float32), 
            shape=(10,), 
            dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(2)] # in future can use different capacities from a list of capacities for each FU]
        
        self.orders = [[],[]]
        self.to_do = np.array([0, 0], dtype=np.int32) # number of orders for each FU
        for i in range(2):
            self.orders[i].append(i + 1) # add order id
            self.orders[i].append(np.random.randint(MIN_ORDERS, MAX_ORDERS + 1)) # add number in order
            self.orders[i].append(i * np.random.randint(0, LATEST_ORDER_START + 1)) # add start date
            self.orders[i].append(np.random.randint(self.orders[i][2] + max_time*(self.orders[i][1]+1), LATEST_ORDER_DUE + 1)) # add due date
            self.to_do[i] += self.orders[i][1] # update total number of orders for each FU
        self.orders = np.array(self.orders, dtype=np.float32)
        self.complete = np.array([0, 0], dtype=np.int32) # number of completed orders for each FU
        self.busy = np.array([0, 0], dtype=np.int32) # busy status of each FU

    def summary(self):
        avg_waiting_times = []
        avg_system_times = []
        total_working_times = []
        obs = np.array([], dtype=np.float32)
        obs = np.append(obs, [self.to_do, self.complete])
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