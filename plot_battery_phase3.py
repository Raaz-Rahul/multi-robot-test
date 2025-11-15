# plot_phase3_results.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_monitor(monitor_csv="monitor.csv"):
    if not os.path.exists(monitor_csv):
        print("monitor.csv not found:", monitor_csv)
        return
    df = pd.read_csv(monitor_csv, comment='#')
    plt.figure(figsize=(10,4))
    plt.plot(df['r'].rolling(5, min_periods=1).mean(), label="Episode reward (smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Learning Curve")
    plt.grid()
    plt.legend()
    plt.show()

def plot_battery_from_array(battery_array):
    battery_array = np.array(battery_array)
    plt.figure(figsize=(10,4))
    for i in range(battery_array.shape[1]):
        plt.plot(battery_array[:, i], label=f"Robot {i}")
    plt.xlabel("Step")
    plt.ylabel("Battery")
    plt.title("Battery Consumption")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_monitor()
