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
    def __init__(self):
            self.arrival_times = [[],[]]
            self.start_service_times = [[],[]]
            self.departure_times = [[],[]]
            
            self.tool = FU[FU]
                
                self.arrival_times[tool].append(arrival)
                self.start_service_times[tool].append(start_job)
                self.departure_times[tool].append(departure)

            def summary(self):
                results = {}
                waiting_times = []
                system_times = []
                no_jobs = 0
                for i in range(len(self.arrival_times)):
                    n = len(self.arrival_times[i])
                    no_jobs += n
                    if n == 0:
                        return {}
                    waiting_times.append([s - a for a, s in zip(self.arrival_times[i], self.start_service_times[i])])
                    system_times.append([d - a for a, d in zip(self.arrival_times[i], self.departure_times[i])])

                return {
                    "number_jobs": no_jobs,
                    "avg_wait_time": sum([sum(wt) for wt in waiting_times]) / no_jobs,
                    "avg_system_time": sum([sum(st) for st in system_times]) / no_jobs,
                    "max_wait_time": max([max(wt) for wt in waiting_times]),
                    "max_system_time": max([max(st) for st in system_times]),
                }
        super().__init__()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Example: 3 actions, 0: hold, 1: request to start A, 2: request to start B
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=float)  # Example: 4D observation

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.env = simpy.Environment()            
        # Run environment
        random.seed(RANDOM_SEED)
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
    
    def step(self, action):
        # let simpy run the environment and define rewards
        with self.machine.request() as req:
                yield req
                start_service_time = env.now
        return obs, reward, terminated, truncated, info


