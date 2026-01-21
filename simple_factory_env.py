import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

class SimpleFactoryEnv(gym.Env):
    def __init__(self, order, max_steps=100):
        super().__init__()
        self.A_time = 4
        self.B_time = 3
        self.order = order
        self.busy = np.array([0,0], dtype=np.int32)
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)
        self.action_space = gym.spaces.Discrete(3)              # 3 potential actions, 1 for move or 0 for hold
        self.observation_space = gym.Spaces.Box(
            low=np.array([0,0,0.0,0.0]), 
            high=np.array([1,1,self.A_time, self.B_time]), shape=(4,), dtype=np.float32
        )
        self.current_state = np.array([self.busy, self.busy, self.A_time, self.B_time]) # flat vector makes proccessing easier
        self.to_do = None
        self.max_steps = max_steps
        self.step_count = 0
        self.to_do = order
        self.doing = 0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.A_time = 4
        self.B_time = 3
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)

        self.busy = np.array([0,0], dtype=np.int32)
        self.current_state = np.array([self.busy, self.time_remaining])
        obs = self.current_state.flatten().astype(float)
        self.step_count = 0
        self.next_state = None
        self.to_do = len(self.order) # number of orders
        self.doing = 0
        return obs, {}


    def step(self, action):
        done = False
        truncated = False                       # if ran out of steps then end episode

        if self.step_count >= self.max_steps:
            truncated = True
            obs = self.current_state.flatten().astype(float)
            return obs, reward, done, truncated, {}
        elif self.to_do == 0 and self.doing == 0:     # if there are no more orders to complete then done and reward
            reward = 100
            done = True
            obs = self.current_state.flatten().astype(float)
            return obs, reward, done, truncated, {}
        
        if action == 0:
            self.time_remaining = -1
            self.next_state = np.array([self.busy, self.time_remaining])

        elif action == 1:
            self.next_state = np.array([[1,1],[self.A_time,self.B_time]])
            if self.to_do > 0:
                self.doing +=1
        # reasons for an invalid move:
        # if the either FUs in current state still has to complete it's action and next state has moved on then negative reward
        if self.current_state[[1,0]] > 0 and self.next_state[[1,0]] == self.A_time \
            or self.current_state[[1,1]] > 0 and self.next_state[[1,1]] == self.B_time \
            :
            reward = -1

        # Reasons for positive rewards:
        # FU finishes a single unit
        elif self.next_state[[1,0]] == 0 or self.next_state[[1,1]] == 0:
            reward = 1
        

        self.current_state = self.next_state
        obs = self.current_state.flatten().astype(float)

        self.step_count += 1
        return obs, reward, done, truncated, {}


