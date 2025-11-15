import numpy as np
import matplotlib.pyplot as plt

def plot_battery(battery_history):
    battery_history = np.array(battery_history)

    plt.figure(figsize=(10,5))
    for i in range(battery_history.shape[1]):
        plt.plot(battery_history[:, i], label=f"Robot {i}")

    plt.xlabel("Time Step")
    plt.ylabel("Battery Level")
    plt.title("Battery Consumption Curve (Phase 3)")
    plt.legend()
    plt.grid(True)
    plt.show()
