import random
import numpy as np
import matplotlib.pyplot as plt

GridState = tuple[int, int]  # (row, col) position in the grid

end = (2, 2)  # goal position   

Q: dict[tuple[int, int, int], float] = {} # Empty Q table containing state values for x and y coordinates and action



def get_Q(Q, state, action):
    return Q.get((state[0], state[1], action),0.0)

#Q[(state[0],state[1], action)] = new_value

def choose_action(state, Q, epsilon:float) -> int:
    if random.random() < epsilon:   # takes a random action not based on Q values
        #print("Exploring")
        return random.randint(0,3)
    else:                           # takes the action based on which action at that state has the highest Q value
        best_a = 0
        best_q = get_Q(Q, state, 0) # initialises Q value using action of 0 for that state
        for a in range(1,4):        # checks all actions to see which has the highest Q value for that state
            q = get_Q(Q, state, a)
            if q > best_q:
                best_q = q
                best_a = a
        #print("Exploiting")
        return best_a

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
    if next_state[0] < 0 or next_state[0] > 2 or next_state[1] < 0 or next_state[1] > 2 or next_state == (1,1) or next_state == (0,2):
        next_state = state  # invalid move, stay in place
        reward = -30.0
        done = True
        return next_state, reward, done
    elif next_state == end:
        reward = 50
        done = True
        return next_state, reward, done
    else:
        reward = 1
        done = False
        return next_state, reward, done

alpha = 0.1                         # learning rate
gamma = 0.9                         # discount factor   
epsilon_min = 0.1                     # minimum exploration rate
epsilon_start = 0.99                   # starting exploration rate
epsilon_decay = 0.01  # decay rate for exploration  
steps = []  # to record steps per episode
ended = []
for episode in range(1500):
    state = (0,0)
    done = False
    i = 0   
    epsilon = max(epsilon_min, epsilon_start**(epsilon_decay*episode))  # decaying epsilon
    while not done:
        action = choose_action(state, Q, epsilon)
        next_state, reward, done = step(state, action)
        #print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        i+=1
        max_next = max(get_Q(Q, next_state, a) for a in range(4))

        Q[state[0], state[1], action] = get_Q(Q, state, action) + alpha * (reward + gamma * max_next - get_Q(Q, state, action))
        state = next_state
        if state == end:
            ended.append(episode)
    steps.append([episode, i])
print(steps)
max_Q_key = max(Q, key=Q.get)
max_Q_value = Q[max_Q_key]
print(max_Q_key, max_Q_value)
print(get_Q(Q, (0,1), 1), get_Q(Q, (0,1), 3), get_Q(Q, (1,0), 3))
print(get_Q(Q, (0,1), 3))
print(ended)

Q_values = []
for i in range(3):
    rows = []
    for j in range(3):
        cols = []
        for a in range(4):
            cols.append(get_Q(Q, (i,j), a))

        rows.append(cols)
    Q_values.append(rows)
print(Q_values)

state = (0,0)
done = False
path = []
while not done:
    action = choose_action(state, Q, 0)
    next_state, reward, done = step(state, action)
    path.append((state))
    state = next_state
path.append(end)
print("Chosen Path:", path)

step_counter = np.array(steps)
x = step_counter[:,0]
y = step_counter[:,1]
plt.plot(x, y, marker='o')
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
plt.show()