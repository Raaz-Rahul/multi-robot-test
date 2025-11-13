# test_phase3_eval.py
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/ppo_multi_robot_coop.zip"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/ppo_multi_robot_coop"  # fallback

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("PPO model not found. Run training first.")

model = PPO.load(MODEL_PATH)
env = MultiRobotEnergyEnvPhase3(debug=True)

NUM_EPISODES = 5
total_rewards = []
battery_logs = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total = 0.0
    btrace = []
    step = 0
    print(f"\n=== Episode {ep+1} ===")
    while not done and step < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action.tolist())
        total += reward
        states = obs.reshape(env.n_robots, 4)
        btrace.append(states[:, 3].tolist())
        step += 1
    total_rewards.append(total)
    battery_logs.append(np.array(btrace))
    print(f"Episode {ep+1} total reward: {total:.2f}")

print("\nAverage reward:", np.mean(total_rewards))

# Plot battery traces of first episode
if battery_logs:
    b0 = battery_logs[0]
    plt.figure(figsize=(8,4))
    for i in range(env.n_robots):
        plt.plot(b0[:, i], label=f"Robot {i}")
    plt.xlabel("Step")
    plt.ylabel("Battery")
    plt.title("Battery trace (ep1)")
    plt.legend()
    plt.grid(True)
    plt.show()
