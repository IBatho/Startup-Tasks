import random
import numpy as np
import matplotlib.pyplot as plt

GridState = tuple[int, int]  # (row, col) position in the grid

end = (5, 5)  # goal position 
grid = (6,6)  # grid size 6x6 grid becasue 0 is start
obstacles = {(1,1), (0,2), (3,3)}  # positions of obstacles

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
    if next_state[0] < 0 or next_state[0] >= grid[0] or next_state[1] < 0 or next_state[1] >= grid[1] or next_state in obstacles:
        next_state = state  # invalid move, stay in place
        reward = -200
        done = True
        return next_state, reward, done
    elif next_state == end:
        reward = 70
        done = True
        return next_state, reward, done
    else:
        reward = 13
        done = False
        return next_state, reward, done

alpha = 0.3                         # learning rate
gamma = 0.3 #{1: 0.1, 2: 0.3, 3: 0.5, 4: 0.9}                        # discount factor   
epsilon_min = 0.1                     # minimum exploration rate
epsilon_start = 0.99                   # starting exploration rate
epsilon_decay = 0.01  # decay rate for exploration  
steps = []  # to record steps per episode
ended = []
average_steps = []
for run in range(1,5):  # run the training multiple times to see different results
    total_steps = 0
    #gamma_value = gamma[run]
    for episode in range(4000):
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
        total_steps += i
        steps.append([episode, i])
    #average_steps.append([gamma_value,(total_steps / 1500)])
    #print(steps)
    max_Q_key = max(Q, key=Q.get)
    max_Q_value = Q[max_Q_key]
    #print(max_Q_key, max_Q_value)
    #print(get_Q(Q, (0,1), 1), get_Q(Q, (0,1), 3), get_Q(Q, (1,0), 3))
    #print(get_Q(Q, (0,1), 3))
    print(ended)

Q_values = []
for i in range(grid[0]):
    rows = []
    for j in range(grid[1]):
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
    if len(path) > 500:  # prevent infinite loops
        print("Failed to find path within 200 steps")
        break
path.append(end)
print("Chosen Path:", path)

'''step_counter = np.array(average_steps)
x = step_counter[:,0]
y = step_counter[:,1]
plt.plot(x, y, marker='o')
plt.title('Average steps for various discount rates (gamma)')
plt.xlabel('Discount Rate (gamma)')
plt.ylabel('Average Total Steps')
plt.grid(True)
plt.show()'''

action_to_arrow = {0: "↑", 1: "↓", 2: "←", 3: "→"}
for i in range(grid[0]):
    row_str = ""
    for j in range(grid[1]):
        s = (i, j)
        if s in obstacles:
            cell = "X"
        elif s == end:
            cell = "G"
        else:
            # greedy action
            best_a = max(range(4), key=lambda a: get_Q(Q, s, a))
            cell = action_to_arrow[best_a]
        row_str += cell + " "
    print(row_str)

action_vec = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right

Xs, Ys, Us, Vs = [], [], [], []
for i in range(grid[0]):
    for j in range(grid[1]):
        s = (i,j)
        if s in obstacles or s == end:
            continue
        a = max(range(4), key=lambda act: get_Q(Q, s, act))
        dx, dy = action_vec[a]
        Xs.append(j);    Ys.append(i)      # no flip
        Us.append(dx);  Vs.append(dy)

plt.figure()
plt.gca().invert_yaxis()                  # make row 0 at top
plt.quiver(Xs, Ys, Us, Vs, angles='xy', scale_units='xy', scale=1)
for (i,j) in obstacles:
    plt.scatter(j, i, c='red', s=80)
plt.scatter(end[1], end[0], c='green', s=80)
plt.xlim(-0.5, grid[1]-0.5); plt.ylim(-0.5, grid[0]-0.5)
plt.gca().set_aspect('equal'); plt.grid(True)
plt.show()

values = np.zeros((grid[0], grid[1]))

for i in range(grid[0]):
    for j in range(grid[1]):
        s = (i, j)
        if s in obstacles:
            values[i, j] = np.nan   # mask obstacles
        else:
            values[i, j] = max(get_Q(Q, s, a) for a in range(4))

plt.imshow(values, cmap='viridis', origin='lower', interpolation='nearest')
plt.colorbar(label='V(s) = max_a Q(s,a)')
plt.show()