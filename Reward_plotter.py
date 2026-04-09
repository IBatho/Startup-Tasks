import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

data = pd.read_csv("PPO_29_reward_data.csv")
x = data["Step"]
y = data["Value"]
# coefs = np.polyfit(x, y, deg=3)
# trendline = np.poly1d(coefs)
trendline_y = y.rolling(window=10, min_periods=1).mean()
plt.figure(figsize=(10, 6))
plt.scatter(x,y,color="red", alpha = 0.8, label="Raw Data", s=30)
plt.plot(x, trendline_y, color="steelblue", linewidth=4, label="Trendline")
plt.legend(
    fontsize=25,          # Increases text size
    title_fontsize=25,    # Increases title size
    loc='lower right',     # Moves it so it doesn't overlap data
    labelspacing=1.2      # Adds vertical space between entries
)
plt.xlabel("Time-steps", fontsize=30, fontweight="bold")
plt.ylabel("Mean Episode Reward", fontsize=30, fontweight="bold")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("PPO: Graph of Mean Episode Reward During Model Training", fontsize=30, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()