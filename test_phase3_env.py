import numpy as np
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

from thesis_plots_phase3 import (
    plot_battery_thesis,
    plot_reward_thesis,
    plot_robot_trajectories
)


# ============================================================
#  Test Model on Phase 3 Cooperative Environment
# ============================================================

def run_phase3_evaluation(mmodel_path="models/ppo_cooperative_phase3"",
                          episodes=5,
                          max_steps=400,
                          debug=True):

    print("\n================ PHASE 3 EVALUATION ================\n")

    # Load environment
    env = MultiRobotEnergyEnvPhase3(debug=debug)

    # Load PPO Model
    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path)

    all_rewards = []

    for ep in range(1, episodes + 1):

        obs, info = env.reset()

        battery_log = []
        reward_log = []
        position_log = [[] for _ in range(env.n_robots)]

        transfers = []
        deliveries = []

        total_reward = 0

        for step in range(max_steps):

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, trunc, info = env.step(action.tolist())

            obs_matrix = obs.reshape(env.n_robots, 5)

            # Save logs
            battery_log.append(obs_matrix[:, 3])
            reward_log.append(reward)

            for i in range(env.n_robots):
                position_log[i].append([obs_matrix[i, 0], obs_matrix[i, 1]])

            # Record events
            if "transfers" in info and info["transfers"]:
                transfers.append((step, "transfers", info["transfers"]))

            if "deliveries" in info and info["deliveries"]:
                deliveries.append((step, "deliveries", info["deliveries"]))

            total_reward += reward

            if done:
                break

        print(f"\n=== Episode {ep} finished in {step} steps | reward {total_reward:.2f} ===")
        print(*deliveries)
        print(*transfers)

        all_rewards.append(total_reward)

        # Convert logs to arrays for plotting
        battery_log = np.array(battery_log)

        # Thesis-ready figures
        print("Generating thesis figures...")
        plot_battery_thesis(env, battery_log)
        plot_reward_thesis(reward_log)
        plot_robot_trajectories(env, position_log)

    print("\n================ EVALUATION COMPLETE ================\n")
    print(f"Average Episode Reward = {np.mean(all_rewards):.2f}")

    return all_rewards


if __name__ == "__main__":
    run_phase3_evaluation()
