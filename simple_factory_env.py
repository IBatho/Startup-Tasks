import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

class SimpleFactoryEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.busy = busy
        self.A_time = 4
        self.B_time = 3
        self.busy = np.array([0,0], dtype=np.int32)
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.Spaces.Box(
            low=np.array([[0,0],[0.0,0.0]]), 
            high=np.array([[1,1],[self.A_time, self.B_time]]), shape=(2,)
        )
        self.current_state = np.array([[self.busy, self.busy],[self.A_time, self.B_time]])
        self.to_do = None
        self.doing = None
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.A_time = 4
        self.B_time = 3
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)

        self.busy = np.array([0,0], dtype=np.int32)
        self.current_state = np.array([self.busy,self.time_remaining])
        obs = self.current_state
        self.step_count = 0
        return obs, {}


    def step(self, action):
        done = False
        truncated = False                       # if ran out of steps then end episode

        if self.step_count >= self.max_steps:
            truncated = True
        elif to_do = None and doing = None:     # if there are no more orders to complete then done and reward
            reward += 1
            done = True
        else:

        return obs, reward, done, truncated, {}


