# test_phase3_env.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

def evaluate(model_path="models/ppo_phase3", episodes=3, max_steps=400, debug=True):
    print("Loading model:", model_path)
    model = PPO.load(model_path)

    env = MultiRobotEnergyEnvPhase3(debug=debug)

    for ep in range(1, episodes+1):
        obs, _ = env.reset()
        done = False
        t = 0
        rewards = []
        battery_history = []
        deliveries = []
        transfers = []

        print(f"\n===== Episode {ep} =====")
        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action.tolist())

            rewards.append(reward)
            states = obs.reshape(env.n_robots, 5)
            battery_history.append(states[:, 3].copy())

            if info.get("deliveries"):
                deliveries.extend(info["deliveries"])
            if info.get("transfers"):
                transfers.extend(info["transfers"])

            print(f"Step {t} | Action={action.tolist()} | Reward={reward:.2f} | Done={done}")
            print(f"Step info: transfers={info.get('transfers',[])}, deliveries={info.get('deliveries',[])}")
            t += 1

        total = sum(rewards)
        print(f"Episode {ep} finished in {t} steps | Total reward = {total:.2f}")
        battery_history = np.array(battery_history)

        # battery plot
        plt.figure(figsize=(8,4))
        for i in range(env.n_robots):
            plt.plot(battery_history[:, i], label=f"Robot {i}", linewidth=2)
        plt.title(f"Battery Trace — Episode {ep}")
        plt.xlabel("Step")
        plt.ylabel("Battery")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate()
