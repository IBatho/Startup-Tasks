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
        self.action_space = gym.spaces.Discrete(2)              # 3 potential actions, 1 for move or 0 for hold
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0.0,0.0]), 
            high=np.array([1, 1, self.A_time, self.B_time]), shape=(4,), dtype=np.float32
        ) # flat vector makes proccessing easier
        self.current_state = np.array([self.busy, self.time_remaining]) 
        self.to_do = None
        self.max_steps = max_steps
        self.step_count = 0
        self.to_do = self.order
        self.doing = None 
        self.complete = None
        self.next_state = self.current_state


        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.order = np.random.randint(0, 101)
        self.A_time = 4
        self.B_time = 3
        self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)

        self.busy = np.array([0,0], dtype=np.int32)
        self.current_state = np.array([self.busy, self.time_remaining])
        obs = self.current_state.flatten().astype(float)
        self.step_count = 0
        self.next_state = self.current_state
        self.to_do = self.order # number of orders
        self.doing = 0
        self.complete = 0
        return obs, {}


    def step(self, action):
        terminated = False
        truncated = False 
        reward = 0
        a_current_busy = self.busy[0]
        b_current_busy = self.busy[1]
                             # if ran out of steps then end episode

        if self.step_count >= self.max_steps:
            truncated = True
            obs = self.current_state.flatten().astype(float)
            reward = -1
            return obs, reward, terminated, truncated, {}
        # if there are no more orders to complete then done and reward
        elif self.to_do == 0 and self.doing == 0:     
            reward = 200
            terminated = True
            obs = self.current_state.flatten().astype(float)
            return obs, reward, terminated, truncated, {}
        
        # If action is to stay then decrease time remaining
        if action == 0:
            if self.busy[0] == 1:
                self.time_remaining[0] -= 1
            if self.busy[1] == 1:
                self.time_remaining[1] -= 1
            self.next_state = np.array([self.busy, self.time_remaining])

        # If action is to move: reset FU times remaining at end
        elif action == 1:

            
            # If A (FU before another FU) is not currently busy (can be made into loop for more FUs) 
            # and an order is added the next state will only have A as busy and B will be free e.g. at the start of order
            if self.busy[0] == 0:
                self.busy[0] += 1

            # If no more orders to complete and units move to next FU so A won't be busy and B will be
            elif self.to_do == 0:
                self.busy[0] = 0
                self.busy[1] += 1

            else: 
                #increase busyness by 1 becasue order is added
                self.busy += 1

            # If the time remaining to completion of the current state is less than or equal to 0 then FU is done with item and can pass on
            # to next FU or complete zone so busyness decreases. less then 0 because item may be held in FU after complete.
            # Taking away 1 because added before for new item but if this item can move it reduces the business
            if self.time_remaining[0] <= 0:
                self.busy[0] -= 1

            if self.time_remaining[1] <= 0:
                self.busy[1] -= 1
                self.doing -= 1
                self.complete += 1

            if self.to_do > 0:
                self.doing +=1
                self.to_do -= 1
            else:
                print('complete all orders')

            self.time_remaining = np.array([self.A_time, self.B_time], dtype=np.int32)
        
            self.next_state = np.array([self.busy, self.time_remaining])
        


        # reasons for an invalid move:
        # if the either FUs have more FU than allowed (1) then negative reward and terminate. This means due to previous code that item 
        # didn't complete it's time in the FU.
        if self.busy[0] > 1 or self.busy[1] > 1:
            reward = -5
            terminated = True

        # Reasons for positive rewards:
        # FU finishes a single unit
        elif self.current_state[0][0] == 0 or self.current_state[0][1] == 0 and self.busy[0] == 1:
            reward = 40
        elif self.time_remaining[0] < 0 or self.time_remaining[1] < 0:
            reward = self.time_remaining 

        self.current_state = self.next_state
    
        obs = self.current_state.flatten().astype(float)

        self.step_count += 1
        print(f"Current State: {self.current_state}, Action: {action}, Steps: {self.step_count}, To Do: {self.to_do}, \
              Doing: {self.doing}, Complete: {self.complete}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return obs, reward, terminated, truncated, {}


