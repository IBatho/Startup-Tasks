import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

class SimpleFactoryEnv(gym.Env):
    def __init__(self, max_steps=100):
        super().__init__()
        self.A_duration = 4
        self.B_duration = 3
        self.max_steps = max_steps

        self.order = None
        self.action_space = gym.spaces.Discrete(2)              # 3 potential actions, 1 for move or 0 for hold
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0.0,0.0]), 
            high=np.array([1, 1, self.A_duration*1.5, self.B_duration*1.5]), shape=(4,), dtype=np.float32
        ) # flat vector makes proccessing easier

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_orders = np.random.randint(5, 16)
        self.to_do = self.total_orders # number of orders

        self.A_time = 0 # time A has been working
        self.B_time = 0 # time B has been working
        self.A_busy = 0
        self.B_busy = 0
        self.step_count = 0
        self.complete = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = self.current_state.flatten().astype(float)
        return obs
    
    def _get_busy(self):
        self.busy = np.array([self.A_busy, self.B_busy], dtype=np.int32)
        return self.busy
    
    def _get_time(self):
        self.time_working = np.array([self.A_time, self.B_time], dtype=np.float32)
        return self.time_working
    
    def _get_state(self):
        self.current_state = np.array([self._get_busy(), self._get_time()])
        return self.current_state

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = False 
        reward = 0
        # if there are no more orders to complete then done and reward
        if self.to_do == 0 and self.doing == 0:     
            reward += 200
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}
        # if ran out of steps then end episode
        elif self.step_count >= self.max_steps:
            truncated = True
            reward -= 1
            return self._get_obs(), reward, terminated, truncated, {}

        
        # If action is to stay then decrease time remaining
        # If the time working is longer than neccessary decrease the reward
        if action == 0:
            if self.A_busy == 1:
                self.A_time += 1
                if self.A_time > self.A_duration:
                    reward -= 0.5

            if self.B_busy == 1:
                self.B_time += 1
                if self.B_time > self.B_duration:
                    reward -= 0.5

            self.next_state = self._get_state()

        # If action is to move: reset FU times remaining at end
        elif action == 1:
        # Taking away 1 because added before for new item but if this item can move it reduces the busyness
        # if the either FUs still have time to work then negative reward and terminate. This means due to previous code that item 
        # didn't complete it's time in the FU.
        # Check B first
            if self.B_busy == 1 and self.B_time >= self.B_duration:
                # check if B has completed time
                self.B_busy -= 1
                self.B_time = 0
                self.complete += 1
                reward += 40
            else:
                reward -= 10
                terminated = True
                return self._get_obs(), reward, terminated, truncated, {}

            # check A
            if self.A_busy == 1:
                # Check if A has completed working
                if self.A_time >= self.A_duration:
                    # check if B is free to move on to
                    if self.B_busy == 0:
                        self.A_busy -= 1
                        self.B_busy += 1
                        self.complete += 1
                        reward += 1
                    else:
                        reward -= 1
                        terminated = True
                else:
                    reward -= 10
                    terminated = True
                    return self._get_obs(), reward, terminated, truncated, {}

            # If A (FU before another FU) is not currently busy (can be made into loop for more FUs) 
            # and an order is added the next state will only have A as busy and B will be free e.g. at the start of order
            if self.A_busy == 0 and self.to_do > 0:
                self.A_busy += 1
                self.A_time = 0
                self.to_do -= 1
                self.next_state = self._get_state()

            # If no more orders to complete and units move to next FU so A won't be busy and B will be
            elif self.to_do == 0:
                self.B_busy += 1
            else: 
                #increase busyness by 1 becasue order is added
                self.A_busy += 1
                self.B_busy += 1

            if self.to_do > 0:
                self.to_do -= 1
            elif self.complete == self.total_orders:
                print('complete all orders')

            self.next_state = self._get_state()
        self.current_state = self.next_state
        self.doing = self.A_busy + self.B_busy
        print(f"Current State: {self.current_state}, Action: {action}, Steps: {self.step_count}, To Do: {self.to_do}, \
              Doing: {self.doing}, Complete: {self.complete}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return self._get_obs(), reward, terminated, truncated, {}