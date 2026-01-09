import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1") #render_mode="human")


all_ep_info = []
for episode in range(20):
    start = time.perf_counter()
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    while not done:
        action = env.action_space.sample() # random action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward
    end = time.perf_counter()
    timer = end - start
    ep_info = [episode +1,timer,total_reward, step_count]
    all_ep_info.append(ep_info)  # episode, time, total reward, steps

   # print(f"Episode {episode + 1} finished in {timer:.4f} seconds with reward:{total_reward} and steps:{step_count}")
final_info = np.array(all_ep_info)
print(final_info)

x = final_info[:,0]
y = final_info[:,3]
plt.plot(x, y, marker='o')
plt.title('CartPole-v1: Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
plt.show()
env.close()