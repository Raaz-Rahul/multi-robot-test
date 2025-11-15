import pandas as pd
import matplotlib.pyplot as plt

def plot_reward_csv(path="monitor.csv"):
    df = pd.read_csv(path, skiprows=1)
    plt.figure(figsize=(10,4))
    plt.plot(df["r"], label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Phase 3 PPO Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
