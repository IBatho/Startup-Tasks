import gymnasium as gym

GridState = tuple[int, int]  # (row, col) position in the grid

def step(state: GridState, action: int) -> tuple[GridState, float,bool]:
    actions = {
        0: (-1, 0),  # up
        1: (1, 0),   # down 
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    return