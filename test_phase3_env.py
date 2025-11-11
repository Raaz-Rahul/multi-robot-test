"""
test_phase3_env.py
Evaluate trained Phase 3 cooperative PPO model in the multi-robot environment.
Author: Rahul Kapardar
"""

from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load environment and model
env = MultiRobotEnergyEnvPhase3(debug=True)
model_path = "models/ppo_multi_robot_coop.zip"

try:
    model = PPO.load(model_path)
    print(f"✅ Loaded PPO model from {model_path}")
except:
    print("❌ PPO model not found. Please train it first using master_train_phase3.py.")
    raise SystemExit

# ---------------------------------------------------------
# Run evaluation
NUM_EPISODES = 10
battery_logs = []
total_rewards = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done, step, total_reward = False, 0, 0
    battery_trace = []

    print(f"\n🚀 Starting Episode {ep + 1}")

    while not done and step < 50:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action.tolist())
        total_reward += reward

        # log battery levels
        states = obs.reshape(env.n_robots, 4)
        battery_trace.append(states[:, 3].tolist())

        step += 1

    total_rewards.append(total_reward)
    battery_logs.append(battery_trace)
    print(f"🏁 Episode {ep + 1} finished with Total Reward = {total_reward:.2f}")

# ---------------------------------------------------------
# Print average reward
print("\n✅ Evaluation complete!")
print(f"Average Episode Reward: {np.mean(total_rewards):.2f}")

# ---------------------------------------------------------
# Visualization Section
try:
    plt.figure(figsize=(10, 5))
    for i in range(env.n_robots):
        plt.plot([b[i] for b in battery_logs[0]], label=f"Robot {i} Battery")
    plt.xlabel("Time Steps")
    plt.ylabel("Battery Level")
    plt.title("Battery Consumption per Robot (Sample Episode)")
    plt.legend()
    plt.grid(True)
    plt.show()
except Exception as e:
    print("⚠️ Plot skipped:", e)

# ---------------------------------------------------------
# Reward distribution plot
try:
    plt.figure(figsize=(8, 4))
    plt.plot(total_rewards, marker='o')
    plt.title("Episode-wise Total Rewards (Phase 3 PPO Evaluation)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
except Exception as e:
    print("⚠️ Reward plot skipped:", e)
