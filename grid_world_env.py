import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        self.fig, self.ax = None, None

        self.action_space = spaces.Discrete(4) # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(
            low=np.array([0,0]),
            high = np.array([self.grid_size[0]-1, self.grid_size[1]-1]),
            dtype=np.int32
        )
        self.current_state = self.start
        render_mode = render_mode
    
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
        if isinstance(action, np.ndarray):
            action = int(action.item())
        actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down 
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        movement = actions[action]
        next_state = (self.current_state[0] + movement[0], self.current_state[1] + movement[1])
        print(next_state)
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
        if self.render_mode != "human":
            print(f"Agent at {self.current_state}, goal {self.end}, obstacles {self.obstacles}")
            return
        
        rows, cols = self.grid_size
        grid = np.zeros((rows, cols), dtype=int)
        for (r, c) in self.obstacles:
            grid[r][c] = 1  # obstacle

        gr, gc = self.end
        grid[gr][gc] = 2  # goal

        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_xticks(np.arange(cols) + 0.5)
            self.ax.set_yticks(np.arange(rows) + 0.5)
            self.ax.set_xticklabels(range(1, cols + 1))
            self.ax.set_yticklabels(range(1, rows + 1))
            self.ax.grid(color='black', linewidth=1)
            self.ax.invert_yaxis()
            plt.ion()

        self.ax.clear()
        cmap = mcolors.ListedColormap(['white', 'red', 'lightgrey'])
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        self.ax.imshow(grid, cmap=cmap, norm=norm, origin='lower')

        gr, gc = self.end
        self.ax.text(gc, gr, 'X', ha='center', va='center', color='black', fontsize=20)

        ar, ac = self.current_state
        self.ax.scatter([ac], [ar], c='blue', s=100)

        self.ax.set_xticks(np.arange(cols))
        self.ax.set_yticks(np.arange(rows))
        self.ax.grid(which = 'both', color='black', linewidth=1)
        plt.pause(0.001)



        #Ascii rendering
        '''grid = [["." for _ in range(cols)] for _ in range(rows)]

        for (r,c) in self.obstacles:
            grid[r][c] = "O"

        gr, gc = self.end
        grid[gr][gc] = "X"

        ar, ac = self.current_state
        grid[ar][ac] = "A"

        header = "   " + " ".join(str(c+1) for c in range(cols))
      

        for r in reversed(range(rows)):
            row_index = f"{r+1:2d}"
            print(row_index, " ".join(grid[r]))
        print(header)
        print()'''
