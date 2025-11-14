import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

def run_phase3_evaluation():
    
    # -------------------------------
    # 1. MODEL PATH (FIXED)
    # -------------------------------
    model_path = "models/ppo_cooperative_phase3"  
    print(f"Loading PPO model from: {model_path}")

    model = PPO.load(model_path)

    # -------------------------------
    # 2. ENV SETUP
    # -------------------------------
    env = MultiRobotEnergyEnvPhase3(n_robots=3, grid_size=5, battery_init=50, debug=False)
    obs, _ = env.reset()

    episode_rewards = []
    battery_history = []
    done = False
    t = 0

    # -------------------------------
    # 3. RUN 1 EPISODE
    # -------------------------------
    while not done and t < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        episode_rewards.append(reward)

        # Store battery of all robots
        st = obs.reshape(env.n_robots, 4)
        battery_history.append(st[:, 3].copy())

        t += 1

    # -------------------------------
    # 4. RESULTS
    # -------------------------------
    print(f"=== Episode finished in {t} steps | reward {sum(episode_rewards):.2f} ===")

    battery_history = np.array(battery_history)

    # Battery plot
    plt.figure(figsize=(10,4))
    for r in range(env.n_robots):
        plt.plot(battery_history[:, r], label=f"Robot {r}")
    plt.title("Battery Consumption per Robot")
    plt.xlabel("Time Step")
    plt.ylabel("Battery Level")
    plt.legend()
    plt.grid()
    plt.show()

    # Reward curve
    plt.figure(figsize=(10,4))
    plt.plot(episode_rewards)
    plt.title("Reward per Step")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run_phase3_evaluation()
