import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

class number_guesser_env(gym.Env):
    render_mode = ["human"]
    def __init__(self, low=1, high=100, max_steps=100, render_mode=None):
        super().__init__()
        self.low = low
        self.high = high
        self.action_space = gym.spaces.Discrete(high - low +1)
        self.observation_space = gym.spaces.Box(
            low=np.array([self.low, 0], dtype=np.int32),
            high = np.array([self.high, self.high - self.low], dtype=np.int32),
            dtype=np.int32
        )
        self.render_mode = render_mode
        self.target = None
        self.max_steps = max_steps
        self.step_count = 0
        self.last_guess = None
    



    def reset(self, seed=None):
        super().reset(seed=seed)
        self.target = self.np_random.integers(self.low, self.high + 1)
        self.last_guess = self.low
        difference = self.last_guess - self.target
        self.step_count = 0
        obs = np.array([self.last_guess, difference], dtype=np.int32)
        return obs, {}
        

    def step(self, action):
        done = False
        truncated = False
        if self.step_count == self.max_steps:
            truncated = True
            guess = int(action) + self.low
            self.last_guess = guess
            difference = guess - self.target
            obs = np.array([self.last_guess, difference], dtype=np.int32)
            return obs, 0.0, done, truncated, {}
        
        guess = int(action) + self.low
        self.last_guess = guess
        difference = guess - self.target

        if guess == self.target:
            reward = 1
            done = True
        else:
            alpha = 0.1
            reward = float(math.exp(-abs(alpha*difference)))
        
        obs = np.array([self.last_guess, difference], dtype=np.int32)
        self.step_count += 1
        return obs, reward, done, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        print(f"Target: {self.target}, Last Guess: {self.last_guess}")
