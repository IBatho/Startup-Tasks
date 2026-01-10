import gymnasium as gym

GridState = tuple[int, int]  # (row, col) position in the grid

end = (2, 2)  # goal position

def step(state: GridState, action: int) -> tuple[GridState, float,bool]:
    done = False
    if state == (2,2):
        return state, 0.0, True  # already at goal
    start_v = (end[0] - state[0], end[1] - state[1])
    start_s = (start_v[0]**2 + start_v[1]**2)**0.5
    actions = {
        0: (-1, 0),  # up
        1: (1, 0),   # down 
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    movement = actions[action]
    next_state = (state[0] + movement[0], state[1] + movement[1])
    print(next_state)
    if next_state[0] < 0 or next_state[0] > 3 or next_state[1] < 0 or next_state[1] > 3 or next_state == (1,1) or next_state == (0,2):
        next_state = state  # invalid move, stay in place
        reward = -1.0
        done = True
        return next_state, reward, done
    elif next_state == end:
        reward = 1
        done = True
        return next_state, reward, done
    else:
        end_v = (end[0] - next_state[0], end[1] - next_state[1])
        end_s = (end_v[0]**2 + end_v[1]**2)**0.5
        reward = start_s - end_s
        return next_state, reward, done

print(step((0,0), 3))  