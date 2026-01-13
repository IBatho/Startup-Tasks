import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode = None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = (6, 6)
        self.obstacles = {(1, 1), (0, 2), (3, 3)}
        self.start = (0, 0)
        self.end = (5, 5)
        self.max_steps = 100
        self.step_count = 0

        self.action_space = spaces.Discrete(4) # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(
            low=np.array([0,0]),
            high = np.array([self.grid_size[0]-1, self.grid_size[1]-1]),
            dtype=np.int32
        )
        self.current_state = self.start
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start
        self.step_count = 0
        return np.array(self.current_state, dtype=np.int32), {}
    
    def step(self, action):
        done = False
        truncated = False
        if self.current_state == self.end:
            done = True
            return self.current_state, 0.0, done, truncated, {}  # already at goal
        elif  self.step_count >= self.max_steps:
            truncated = True
            return self.current_state, -1.0, done, truncated, {}  # max steps reached
        actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down 
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        movement = actions[action]
        next_state = (self.current_state[0] + movement[0], self.current_state[1] + movement[1])
        if next_state[0] < 0 or next_state[0] >= self.grid_size[0] or \
            next_state[1] < 0 or next_state[1] >= self.grid_size[1] or \
            next_state in self.obstacles:
            next_state = self.current_state  # invalid move, stay in place
            reward = -200
            done = True
        elif next_state == self.end:
            reward = 70
            done = True
        else:
            reward = 13
        self.current_state = next_state # updates the environment's knowledge of the current state
        next_state = np.array(self.current_state, dtype=np.int32)
        self.step_count += 1
        return next_state, reward, done, truncated, {}
    
    def render(self):
        if self.render_mode == "human":
            print(f"Agent at {self.current_state}, goal {self.end}, obstacles {self.obstacles}")