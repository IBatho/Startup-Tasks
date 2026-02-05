import gymnasium as gym
import simpy
import random
import numpy as np
# Get state from simpy environment which includes To do, FU status and time working use this to make action decisions in this 
# gym environment and return the decision to simpy
# Reward for: completing all jobs, completing individual jobs, moving B starting, A starting (not A to B because may request 
# and not move which means held)
# Penalty for: time taken to complete jobs longer than neded, time FU idle, time FU working longer than needed, holding when there are jobs to do

# Envionrment of 2 FUs A and B, where A is the first FU and B is the second FU. 
# Jobs are requested for A and do service time there
# When 1 or 3 requested then move to B when complete.
# Sequential order of using FUs A then B, no skipping or going back.
# Each FU has a capacity of only 1 job at a time and there is no buffer between them, so if B is busy then A cannot move to B and must hold until B is free.

RANDOM_SEED = 42
B_TIME = 5.0
A_TIME = 2.0
FU = {"A":0, "B":1}
NUM_ORDERS = 20

class BasicFactoryEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.FU_times = [A_TIME, B_TIME]
        self.max_steps = max_steps        
        # Define action and observation space
        # Example: 4 actions, 0: hold, 
        # 1: request to start A, 
        # 2: request to start B from A, 
        # 3: request to start both A and B
        self.action_space = gym.spaces.Discrete(4)    
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float32), 
            high=np.array([100,100,1,50,50,100,1,50,50,100], dtype=np.float32), 
            shape=(10,), 
            dtype=np.float32
            )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()  
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(2)] # in future can use different capacities from a list of capacities for each FU]          
        self.job_queue = [] # track pending jobs
        self.total_orders = np.random.randint(5, 16)
        self.to_do = self.total_orders # number of orders
        self.busy = np.array([0, 0], dtype=np.int32) # busy status of each FU
        self.step_count = 0
        self.complete = 0
        self.arrival_times = [[],[]]
        self.start_service_times = [[],[]]
        self.departure_times = [[],[]]
        self.remaining_times = np.array([0.0, 0.0])
        return self.summary(), {}
    

    def summary(self):
        avg_waiting_times = []
        avg_system_times = []
        total_working_times = []
        obs = np.array([], dtype=np.float32)
        no_jobs = 0
        for i in range(len(self.arrival_times)):
            n = len(self.arrival_times[i])
            no_jobs += n
            if n == 0:
                avg_waiting_times.append(0.0)
                avg_system_times.append(0.0)
                total_working_times.append(0.0)
                continue
            else:
                avg_waiting_times.append(np.mean([s - a for a, s in zip(self.arrival_times[i], self.start_service_times[i])]))
                avg_system_times.append(np.mean([d - a for a, d in zip(self.arrival_times[i], self.departure_times[i])]))
                total_working_times.append(sum([d - a for a, d in zip(self.start_service_times[i], self.departure_times[i])]))

        obs = np.append(obs, [self.to_do, self.complete])
        for i in range(len(FU)):
            obs = np.append(obs, [self.busy[i], avg_waiting_times[i], avg_system_times[i], total_working_times[i]])
        return obs

    def job(self, env, name, machine_id):
        # gets the arrival time for that item going into a specific FU
        self.arrival_times[machine_id].append(self.env.now)
        with self.machines[machine_id].request() as req:
            #request the machine to start
            yield req
            if machine_id == 0: # if machine A then set busy A
                self.busy[machine_id] = 1
            else:                       #
                self.busy[machine_id] = 1
                self.busy[machine_id - 1] = 0  # free up A when moving to B
            if machine_id == 0:
                self.to_do -= 1 # change to only A start
            self.start_service_times[machine_id].append(self.env.now)
            service_time = self.remaining_times[machine_id]
            yield env.timeout(service_time)
            self.remaining_times[machine_id] = 0
            self.departure_times[machine_id].append(self.env.now)
            if machine_id == 1:
                self.complete += 1 # change to only B complete
                self.busy[machine_id] = 0 # free up B when complete            

    def step(self, action):
        # let simpy run the environment and define rewards
        terminated = False
        truncated = False
        info = {}
        self.step_count += 1
        reward = 0

        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1
            return self.summary(), reward, terminated, truncated, info
        # in each action set up a job
        if action == 0:
            # hold
            if self.busy[0] == 1 and self.busy[1] == 1 and self.to_do > 0: # if either FU is busy
                reward -= 1  # penalty for holding when FUs are busy
            elif self.busy[0] == 1 and self.busy[1] == 0:
                reward -= 0.5
            elif self.busy[0] == 0 and self.busy[1] == 1:
                reward -= 0.5
        elif action == 1:
            # request to start A
            if self.busy[0] == 0:
                self.remaining_times[0] = self.FU_times[0]
                self.env.process(self.job(self.env, f"Job_A_{self.step_count}", FU["A"]))
        elif action == 2:
            # request to start B, moving from B
            if self.busy[1] == 0 and self.busy[0] == 1 and self.remaining_times[0] == 0: # if B is free and A is busy and A has completed its time
                self.remaining_times[1] = self.FU_times[1]
                self.env.process(self.job(self.env, f"Job_B_{self.step_count}", FU["B"]))
        # generate jobs and run for a time step
        elif action == 3:
            # request to start both A and B
            if self.busy[1] == 0 and self.remaining_times[0] == 0: # if B is free and A has completed its time
                self.remaining_times[0] = self.FU_times[0]
                self.remaining_times[1] = self.FU_times[1]
                self.env.process(self.job(self.env, f"Job_B_{self.step_count}", FU["B"]))
                self.env.process(self.job(self.env, f"Job_A_{self.step_count}", FU["A"]))

        active_times = self.remaining_times[self.remaining_times > 0]
        if len(active_times) > 0:
            service_time = active_times.min()
            self.env.run(until=self.env.now + service_time)  # Run until a machine is complete it's job

        if self.complete == self.total_orders:
            reward = 200
            terminated = True

        return self.summary(), reward, terminated, truncated, info