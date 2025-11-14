# visualize_phase3.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
import os
import glob
import pandas as pd

# 1) Plot BC loss (if present)
if os.path.exists("models/bc_phase3_losses.npz"):
    data = np.load("models/bc_phase3_losses.npz", allow_pickle=True)
    losses = data["losses"]
    plt.figure(figsize=(8,3))
    plt.plot(losses, marker="o")
    plt.title("BC Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
else:
    print("No BC loss file found (models/bc_phase3_losses.npz)")

# 2) Plot PPO learning curve (monitor files)
# Monitor writes files with prefix 'monitor' inside the env folder; search for monitor*.csv
monitor_files = glob.glob("**/monitor.csv", recursive=True) + glob.glob("monitor.csv")
if monitor_files:
    # pick the latest
    mf = sorted(monitor_files)[-1]
    df = pd.read_csv(mf, comment='#')
    plt.figure(figsize=(8,3))
    plt.plot(df['l'].rolling(5).mean(), label='Episode length (smoothed)')
    plt.plot(df['r'].rolling(5).mean(), label='Episode reward (smoothed)')
    plt.title("PPO Learning Curve (smoothed)")
    plt.xlabel("Episode index")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Monitor file used:", mf)
else:
    print("No monitor.csv found. PPO Monitor data missing.")

# 3) Evaluate current PPO model and plot battery traces
MODEL = "models/ppo_phase3.zip"
if not os.path.exists(MODEL) and os.path.exists("models/ppo_phase3"):
    MODEL = "models/ppo_phase3"
if not os.path.exists(MODEL):
    print("PPO model not found at models/ppo_phase3(.zip). Run training first.")
else:
    model = PPO.load(MODEL)
    env = MultiRobotEnergyEnvPhase3(enable_sharing=True, shaped_reward=False, debug=False)

    episodes = 3
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        battery_trace = []
        rewards = []
        steps = 0
        while not done and steps < 400:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action.tolist())
            states = obs.reshape(env.n_robots, 5)
            battery_trace.append(states[:, 3].copy())
            rewards.append(reward)
            steps += 1
        battery_trace = np.array(battery_trace)
        plt.figure(figsize=(8,4))
        for i in range(env.n_robots):
            plt.plot(battery_trace[:, i], label=f"Robot {i}")
        plt.title(f"Battery trace (Episode {ep+1})")
        plt.xlabel("Step")
        plt.ylabel("Battery")
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure(figsize=(8,3))
        plt.plot(rewards)
        plt.title(f"Per-step Reward (Episode {ep+1})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()
