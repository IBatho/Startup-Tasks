import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

class SimpleFactoryEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.A_time = 4
        self.B_time = 3
        self.order = None
        self.busy = np.array([0,0], dtype=np.int32)
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)
        self.action_space = gym.spaces.Discrete(3)              # 3 potential actions, 1 for move or 0 for hold
        self.observation_space = gym.Spaces.Box(
            low=np.array([0,0,0.0,0.0]), 
            high=np.array([1,1,self.A_time, self.B_time]), shape=(4,), dtype=np.float32
        )
        self.current_state = np.array([self.busy, self.busy, self.A_time, self.B_time]) # flat vector makes proccessing easier
        self.to_do = None
        self.max_steps = max_steps
        self.step_count = 0
        self.to_do = self.order
        self.doing = None 
        self.complete = None


        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.order = np.random.random_integers(0, 100)
        self.A_time = 4
        self.B_time = 3
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)

        self.busy = np.array([0,0], dtype=np.int32)
        self.current_state = np.array([self.busy, self.time_remaining])
        obs = self.current_state.flatten().astype(float)
        self.step_count = 0
        self.next_state = None
        self.to_do = self.order # number of orders
        self.doing = 0
        self.complete = 0
        return obs, {}


    def step(self, action):
        done = False
        truncated = False 
        a_busy = self.current_state[0][0]
        b_busy = self.current_state[0][1]
                             # if ran out of steps then end episode

        if self.step_count >= self.max_steps:
            truncated = True
            obs = self.current_state.flatten().astype(float)
            return obs, reward, done, truncated, {}
        elif self.to_do == 0 and self.doing == 0:     # if there are no more orders to complete then done and reward
            reward = 100
            done = True
            obs = self.current_state.flatten().astype(float)
            return obs, reward, done, truncated, {}
        
        # If action is to stay then decrease time remaining
        if action == 0:
            self.time_remaining = -1
            self.next_state = np.array([self.busy, self.time_remaining])

        # If action is to move: reset FU times remaining
        elif action == 1:
            # If the time remaining to completion of the current state is less than or equal to 0 then FU is done and can pass on
            # to next FU so busyness becomes 0 because complet and asked
            if self.time_remaining[0] <= 0 and self.time_remaining[1] <= 0:
                a_busy = 0
                b_busy = 0


            self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)
            # If A (FU before another FU) is not currently busy (can be made into loop for more FUs) 
            # and an order is added the next state will only have A as busy and B will be free
            if a_busy == 0:
                a_busy = 1
                b_busy = 0

            # If no more orders to complete and units move to next FU so A won't be busy and B will be
            elif self.to_do == 0:
                a_busy = 0
                b_busy = +1

            else: 
                #increase busyness by 1 and go back to time to completion
                self.next_state = np.array([[a_busy+1,b_busy+1],[self.A_time,self.B_time]])
            if self.to_do > 0:
                self.doing +=1
                self.to_do -= 1
        

        # reasons for an invalid move:
        # if the either FUs in current state still has to complete it's action and next state has moved on then negative reward
        if self.current_state[[1,0]] > 0 and self.next_state[[1,0]] == self.A_time \
            or self.current_state[[1,1]] > 0 and self.next_state[[1,1]] == self.B_time \
            :
            reward = -1

        # Reasons for positive rewards:
        # FU finishes a single unit
        elif self.next_state[[1,0]] == 0 or self.next_state[[1,1]] == 0:
            reward = 1
        

        self.current_state = self.next_state
        obs = self.current_state.flatten().astype(float)

        self.step_count += 1
        return obs, reward, done, truncated, {}


