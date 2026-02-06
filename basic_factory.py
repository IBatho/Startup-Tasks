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
MIN_ORDERS = 5
MAX_ORDERS = 15


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
            high=np.array([MAX_ORDERS,MAX_ORDERS,1,50,50,1000,1,50,50,1000], dtype=np.float32), 
            shape=(10,), 
            dtype=np.float32
            )


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()  
        self.machines = [simpy.Resource(self.env, capacity=1) for _ in range(2)] # in future can use different capacities from a list of capacities for each FU]          
        self.job_queue = [] # track pending jobs
        self.total_orders = np.random.randint(MIN_ORDERS, MAX_ORDERS + 1)
        self.to_do = self.total_orders # number of orders
        self.busy = np.array([0, 0], dtype=np.int32) # busy status of each FU
        self.step_count = 0
        self.complete = 0
        self.arrival_times = [[],[]]
        self.start_service_times = [[],[]]
        self.departure_times = [[],[]]
        self.remaining_times = np.array([0.0, 0.0])
        self.working = 0

        return self.summary(), {}
    

    def summary(self):
        avg_waiting_times = []
        avg_system_times = []
        total_working_times = []
        obs = np.array([], dtype=np.float32)
        no_jobs = 0
        for i in range(len(self.arrival_times)):
            n_arrivals = len(self.arrival_times[i])
            n_started = len(self.start_service_times[i])
            n_finished = len(self.departure_times[i])
            no_jobs += n_arrivals
            if n_finished > 0:
                avg_waiting_times.append(np.mean([s - a for a, s in zip(self.arrival_times[i], self.start_service_times[i])]))
                avg_system_times.append(np.mean([d - a for a, d in zip(self.arrival_times[i], self.departure_times[i])]))
                total_working_times.append(sum([d - a for a, d in zip(self.start_service_times[i], self.departure_times[i])]))
            else:
                avg_waiting_times.append(0.0)
                avg_system_times.append(0.0)
                total_working_times.append(0.0)
                
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
                self.working += 1
                self.to_do -= 1 # change to only A start

            else:                       #
                self.busy[machine_id] = 1
                self.busy[machine_id - 1] = 0  # free up A when moving to B
            self.reward += 0.5 # reward for starting a job
            self.start_service_times[machine_id].append(self.env.now)
            service_time = self.FU_times[machine_id]
            yield env.timeout(service_time)
            self.departure_times[machine_id].append(self.env.now)
            self.reward += 1 # reward for completing a job at any FU, in future can differentiate between FUs
            if machine_id == len(FU) - 1: # if it's the last FU then reward for completion 
                self.reward += 5
                self.complete += 1
                self.busy[machine_id] = 0
                self.working -= 1

    def step(self, action):
        # let simpy run the environment and define rewards
        terminated = False
        truncated = False
        info = {"total_orders": self.total_orders, 
                "time": self.env.now, 
                "action": action,
                "working": self.working}
        self.step_count += 1
        self.reward = 0

        if self.step_count >= self.max_steps:
            truncated = True
            reward = -1
            return self.summary(), reward, terminated, truncated, info
        # in each action set up a job
        if action == 0:
            # hold
            for i in range(len(FU)):
                if self.remaining_times[i] <= 0 and self.to_do > 0:
                    self.reward -= 0.5 # penalty for holding when there are jobs to do and machine is free
                if self.busy[i] == 1 and self.remaining_times[i] > 0:
                    self.reward +=1 # machine correctly working on a job 

        elif action == 1:
            # request to start A
            if self.busy[0] == 0 and self.to_do > 0: # if A is free and there are jobs to do
                self.remaining_times[0] = self.FU_times[0]
                self.env.process(self.job(self.env, f"Job_A_{self.step_count}", FU["A"]))
            else:
                self.reward -= 0.5 # penalty for requesting to start A when it's busy or there are no jobs to do
        elif action == 2:
            # request to start B, moving from A to B
            if self.busy[1] == 0 and self.busy[0] == 1 and self.remaining_times[0] <= 0 and self.working > 0: # if B is free and A is busy and A has completed its time
                self.remaining_times[1] = self.FU_times[1]
                self.env.process(self.job(self.env, f"Job_B_{self.step_count}", FU["B"]))
            else:
                self.reward -= 0.5 # penalty for requesting to start B when it's busy or A is not ready or there are no jobs to do
        # generate jobs and run for a time step
        elif action == 3:
            # request to start both A and B
            if self.busy[0] == 1 and self.busy[1] == 0 and self.remaining_times[0] <= 0 and self.to_do > 0 and self.working > 0: # if B is free and A has completed its time and their are orders in the system
                self.remaining_times[0] = self.FU_times[0]
                self.remaining_times[1] = self.FU_times[1]
                self.env.process(self.job(self.env, f"Job_B_{self.step_count}", FU["B"]))
                self.env.process(self.job(self.env, f"Job_A_{self.step_count}", FU["A"]))
            else:
                self.reward -= 0.5 # penalty for requesting to start both when not possible to start both
    
        active_times = self.remaining_times[self.remaining_times > 0]
        if len(active_times) > 0: # needs changing to account for if one machine is working and the other is idle, currently just runs until the first machine finishes which may not be correct if the other machine is working on a longer job
            # sevice time should be saved for next step
            # if self.complete == self.total_orders-1 and self.busy[-1] == 1: # if all but one job completed and the last job is currently being worked on then run until it's completion to ensure episode ends with correct reward and state
            #     print("almost done!")
            #     service_time = active_times.min() + 1 # add extra time to ensure all jobs completed are recorded in the state and reward before ending episode
            # else:
            service_time = active_times.min()
            self.env.run(until=self.env.now + service_time)  # Run until a machine is complete it's job
            self.remaining_times -= service_time # may produce negative times but that just indicates how long it's been idle
        elif self.working == 1: # if no machines are working but there are still jobs to do then run for a time step to simulate time passing and potentially add penalty for holding when there are jobs to do
            self.env.run(until=self.env.now + 1) # if no machines are working just run for a time step


        if self.complete == self.total_orders:
            self.reward += 200
            terminated = True
            print("All orders completed!")

        return self.summary(), self.reward, terminated, truncated, info