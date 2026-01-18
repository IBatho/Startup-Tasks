import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class number_guesser_env(gym.Env):
    render_mode = ["human"]
    def __init__(self, low=1, high=100, max_steps=50, render_mode=None):
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
        self.fig, self.ax = None, None
    



    def reset(self, seed=None):
        super().reset(seed=seed)
        self.target = self.np_random.integers(self.low, self.high + 1)
        self.last_guess = self.low
        difference = abs(self.last_guess - self.target)
        self.step_count = 0
        obs = np.array([self.last_guess, difference], dtype=np.int32)
        return obs, {}
        

    def step(self, action):
        done = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            difference = self.last_guess - self.target
            obs = np.array([self.last_guess, difference], dtype=np.int32)
            #print("truncated")
            return obs, 0.0, done, truncated, {}
        
        guess = int(action) + self.low
        self.last_guess = guess
        difference = guess - self.target

        if guess == self.target:
            reward = 100
            done = True
            #print("done")
        else:
            alpha = 0.3

            reward = -10*abs(difference/99) -0.1 #float(-(math.exp(abs(difference))))
        obs = np.array([self.last_guess, difference], dtype=np.int32)
        self.step_count += 1
        #print(obs, reward, done, truncated, {})
        return obs, reward, done, truncated, {}
    
    def render(self):
        if self.render_mode != "human":
            return
        print(f"Target: {self.target}, Last Guess: {self.last_guess}")

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()
        self.ax.clear()
        self.ax.set_xlim(self.low-1, self.high+1)
        self.ax.set_ylim(-1,1)
        self.ax.hlines(0, self.low, self.high, color="black")
        self.ax.set_xticks(range(self.low, self.high + 1))
        self.ax.text(self.target, 0.1, "X", ha = "center", va="bottom", color="black", fontsize=14)

        if self.last_guess is not None:
            self.ax.scatter([self.last_guess], [0], c="red", s=50)

            self.ax.annotate(
                "",
                xy=(self.target,0),
                xytext=(self.last_guess, 0),
                arrowprops=dict(arrowstyle="->", color="blue"),
                )
        plt.pause(0.001)
