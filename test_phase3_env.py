# test_phase3_env.py
"""
Evaluate Phase 3 PPO cooperative model.
Prints deliveries, load-sharing, collisions, battery, rewards.
"""

from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/ppo_multi_robot_coop.zip"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/ppo_multi_robot_coop"  # fallback

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ PPO model not found. Run master_train_phase3.py first.")

print(f"Loading PPO model from: {MODEL_PATH}")
model = PPO.load(MODEL_PATH)

env = MultiRobotEnergyEnvPhase3(debug=True)
EPISODES = 5
total_rewards = []
battery_logs = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    trace = []
    step = 0

    print(f"\n==================== Episode {ep+1} ====================")

    while not done and step < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action.tolist())
        ep_reward += reward

        state_matrix = obs.reshape(env.n_robots, 4)
        trace.append(state_matrix[:, 3])  # battery

        print(f"Step {step} | Action={action.tolist()} | Reward={reward:.2f} | Done={done}")

        step += 1

    total_rewards.append(ep_reward)
    battery_logs.append(np.array(trace))
    print(f"Episode {ep+1} Total Reward = {ep_reward:.2f}")

print("\n==================== Evaluation Complete ====================")
print("Average Episode Reward:", np.mean(total_rewards))

# PLOT BATTERY OF EP1
if battery_logs:
    b0 = battery_logs[0]
    plt.figure(figsize=(8, 4))
    for i in range(env.n_robots):
        plt.plot(b0[:, i], label=f"Robot {i}")
    plt.xlabel("Steps")
    plt.ylabel("Battery")
    plt.title("Battery Consumption — Episode 1")
    plt.legend()
    plt.grid(True)
    plt.show()
