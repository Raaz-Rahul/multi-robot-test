"""
Phase 3 Testing + Visualization Script
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from energy_env_phase3 import MultiRobotEnergyEnvPhase3

def run_phase3_test(n_episodes=5):
    print("\n🚀 Loading PPO Phase-3 Model...")
    model = PPO.load("models/ppo_cooperative_phase3")

    env = MultiRobotEnergyEnvPhase3(debug=True)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        battery_history = []
        deliveries = []
        transfers = []

        while not done and step < 400:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, trunc, info = env.step(action.tolist())

            robot_state = obs.reshape(env.n_robots, 4)
            battery_history.append(robot_state[:, 3])

            if "deliveries" in info and info["deliveries"]:
                deliveries.append((step, info["deliveries"]))

            if "transfers" in info and info["transfers"]:
                transfers.append((step, info["transfers"]))

            obs = next_obs
            step += 1

        print(f"\n=== Episode {ep+1} finished in {step} steps ===")
        print(deliveries)
        print(transfers)

        # Plot energy chart
        battery_history = np.array(battery_history)
        plt.figure(figsize=(10,5))
        for i in range(env.n_robots):
            plt.plot(battery_history[:, i], label=f"Robot {i}")
        plt.legend()
        plt.title(f"Battery Consumption (Episode {ep+1})")
        plt.xlabel("Time Step")
        plt.ylabel("Battery")
        plt.show()


if __name__ == "__main__":
    run_phase3_test()
