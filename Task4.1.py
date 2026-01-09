import random

Q: dict[tuple[int, int, int], float] = {}
def get_Q(Q, state, action):
    return Q.get((state[0], state[1], action),0.0)

Q[(state[0],state[1], action)] = new_value

def choose_action(state, Q, epsilon:float) -> int:
    if random.random() < epsilon:
        return random.randint(0,3)
    else:
        best_a = 
        return best_a
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next - Q[s, a])
