import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import numpy as np

from energy_env_phase3 import MultiRobotEnergyEnvPhase3


def plot_battery(steps=200):

    model = PPO.load("models/ppo_phase3.zip")
    env = MultiRobotEnergyEnvPhase3(debug=False)

    obs, _ = env.reset()

    battery_log = [[] for _ in range(env.n_robots)]

    for t in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action.tolist())

        # extract battery levels
        for i in range(env.n_robots):
            battery_log[i].append(obs[i*4 + 3])

        if done:
            break

    # Plotting
    plt.figure(figsize=(10,5))
    for i in range(env.n_robots):
        plt.plot(battery_log[i], label=f"Robot {i}")

    plt.xlabel("Time Step")
    plt.ylabel("Battery Level")
    plt.title("Robots Battery Consumption Over Time (Phase-3)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_battery()
