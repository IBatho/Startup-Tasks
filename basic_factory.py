import gymnasium as gym
import simpy
import random
import numpy as np
# Get state from simpy environment which includes To do, FU status and time working use this to make action decisions in this 
# gym environment and return the decision to simpy
# Reward for: completing all jobs, completing individual jobs, moving B starting, A starting (not A to B because may request 
# and not move which means held)
# Penalty for: time taken to complete jobs longer than neded, time FU idle, time FU working longer than needed, holding when there are jobs to do

RANDOM_SEED = 42
B_TIME = 5.0
A_TIME = 2.0
FU = {"A":0, "B":1}
NUM_ORDERS = 20

class BasicFactoryEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.FU_times = {0:A_TIME, 1:B_TIME}
        self.max_steps = max_steps        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Example: 3 actions, 0: hold, 1: request to start A, 2: request to start B
        self.observation_space = gym.spaces.Box(low=[0,0,0,0,0,0,0,0,0,0], high=[100,100,1,1,10,10,10,10,10,10], shape=(10,), dtype=float)  # Example: 4D observation


    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()            
        # Run environment
        random.seed(RANDOM_SEED)
        self.total_orders = random.randint(5, 16)
        self.to_do = self.total_orders # number of orders
        self.busy = np.array([0, 0], dtype=np.int32) # busy status of each FU
        self.step_count = 0
        self.complete = 0
        self.arrival_times = [[],[]]
        self.start_service_times = [[],[]]
        self.departure_times = [[],[]]
        return self.summary(), {}
    

    def summary(self):
        waiting_times = []
        system_times = []
        working_times = []
        no_jobs = 0
        for i in range(len(self.arrival_times)):
            n = len(self.arrival_times[i])
            no_jobs += n
            if n == 0:
                waiting_times.append([0])
                system_times.append([0])
                working_times.append(0)
                continue
            waiting_times.append([s - a for a, s in zip(self.arrival_times[i], self.start_service_times[i])])
            system_times.append([d - a for a, d in zip(self.arrival_times[i], self.departure_times[i])])
            working_times.append(sum([d - a for a, d in zip(self.start_service_times[i], self.departure_times[i])]))

        obs = np.array([self.to_do, self.complete, self.busy, waiting_times, system_times, working_times], dtype=np.float32)  
        return obs.flatten().astype(float)

    
    def step(self, action):
        # let simpy run the environment and define rewards
        
    def job(env, name, machine, metrics):
        with self.machine.request() as req:
                yield req
                start_service_time = env.now
        return self.summary(), reward, terminated, truncated, info
    
    def job_generator(env, machine, metrics):
        for i in range(NUM_ORDERS):
            for j in range(len(FU)):
                name = f"Job_{i+1}_FU{j}"
                env.process(job(env, name, machine, metrics))

        self.machine = simpy.Resource(env, capacity=1)
        self.metrics = Metrics()
        self.env.process(job_generator(env, machine, metrics))
        self.env.run()
        summary = metrics.summary()
        print("\nSimulation Summary:")
        for k, v in summary.items():
            print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

