import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

data = pd.read_csv("PPO_29_reward_data.csv")
x = data["Step"]
y = data["Value"]
coefs = np.polyfit(x, y, deg=3)
trendline = np.poly1d(coefs)
plt.figure(figsize=(10, 6))
plt.scatter(x,y,color="red", alpha = 0.8, label="Raw Data", s=10)
plt.plot(x, trendline(x), color="steelblue", linewidth=2, label="Trendline")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel("Mean Episode Reward")
plt.title("PPO: Graph of Mean Episode Reward vs Timesteps")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()