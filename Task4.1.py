import random

GridState = tuple[int, int]  # (row, col) position in the grid

end = (2, 2)  # goal position   

Q: dict[tuple[int, int, int], float] = {}



def get_Q(Q, state, action):
    return Q.get((state[0], state[1], action),0.0)

#Q[(state[0],state[1], action)] = new_value

def choose_action(state, Q, epsilon:float) -> int:
    if random.random() < epsilon:
        print("Exploring")
        return random.randint(0,3)
    else:
        best_a = 0
        best_q = get_Q(Q, state, 0)
        for a in range(1,4):
            q = get_Q(Q, state, a)
            if q > best_q:
                best_q = q
                best_a = a
        print("Exploiting")
        return best_a
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next - Q[s, a])

def step(state: GridState, action: int) -> tuple[GridState, float,bool]:
    if state == end:
        done = True
        return state, 0.0, done  # already at goal
    actions = {
        0: (-1, 0),  # up
        1: (1, 0),   # down 
        2: (0, -1),  # left
        3: (0, 1)    # right
    }
    movement = actions[action]
    next_state = (state[0] + movement[0], state[1] + movement[1])
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
        reward = 1
        done = False
        return next_state, reward, done
        
for episode in range(10):
    state = (0,0)
    done = False
    while not done:
        action = choose_action(state, Q, 0.1)
        next_state, reward, done = step(state, action)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        state = next_state