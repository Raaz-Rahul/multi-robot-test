import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from energy_env_phase3 import MultiRobotEnergyEnvPhase3


# ============================================================
# 1. BATTERY CONSUMPTION PLOT
# ============================================================
def plot_battery_thesis(env, battery_log):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    for i in range(env.n_robots):
        plt.plot(battery_log[:, i], linewidth=2.2, label=f"Robot {i}")

    plt.xlabel("Time Step", fontsize=16)
    plt.ylabel("Battery Level (%)", fontsize=16)
    plt.title("Battery Consumption of Multi-Robot System", fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig("battery_consumption.png", dpi=300)
    plt.show()


# ============================================================
# 2. REWARD CURVE PLOT
# ============================================================
def plot_reward_thesis(reward_log):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})

    # Moving average for smoother curve (window=10)
    window = 10
    smooth_reward = np.convolve(reward_log, np.ones(window)/window, mode='same')

    plt.plot(smooth_reward, color="purple", linewidth=2.5, label="Smoothed Reward")
    plt.xlabel("Time Step", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.title("Reward Curve Over Time", fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()

    plt.savefig("reward_curve.png", dpi=300)
    plt.show()


# ============================================================
# 3. ROBOT TRAJECTORY PLOT (GRID MAP)
# ============================================================
def plot_robot_trajectories(env, position_log):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})

    colors = ["red", "blue", "green", "orange", "purple"]

    for r in range(env.n_robots):
        traj = np.array(position_log[r])
        plt.plot(traj[:, 0], traj[:, 1], color=colors[r], linewidth=2.2, label=f"Robot {r}")

        # Mark start
        plt.scatter(traj[0, 0], traj[0, 1], s=120, color=colors[r], marker="o", edgecolors="black")

        # Mark end
        plt.scatter(traj[-1, 0], traj[-1, 1], s=120, color=colors[r], marker="X", edgecolors="black")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title("Robot Trajectories in Grid Environment", fontsize=18)
    plt.xlabel("X Position", fontsize=16)
    plt.ylabel("Y Position", fontsize=16)
    plt.legend(fontsize=14)
    plt.gca().invert_yaxis()  # Makes visualization intuitive
    plt.tight_layout()

    plt.savefig("robot_trajectories.png", dpi=300)
    plt.show()


# ============================================================
# RUN ONE EPISODE AND LOG EVERYTHING
# ============================================================
def run_and_generate_plots():

    env = MultiRobotEnergyEnvPhase3(debug=False)
    obs, info = env.reset()

    max_steps = 400

    # Logs
    battery_log = []
    reward_log = []
    position_log = [[] for _ in range(env.n_robots)]

    for step in range(max_steps):

        actions = np.random.randint(0, 6, size=env.n_robots)

        obs, reward, done, trunc, info = env.step(actions.tolist())

        obs_matrix = obs.reshape(env.n_robots, 5)
        battery_log.append(obs_matrix[:, 3])
        reward_log.append(reward)

        # Save positions
        for i in range(env.n_robots):
            position_log[i].append([obs_matrix[i, 0], obs_matrix[i, 1]])

        if done:
            break

    battery_log = np.array(battery_log)

    # ===== Generate Thesis Plots =====
    plot_battery_thesis(env, battery_log)
    plot_reward_thesis(reward_log)
    plot_robot_trajectories(env, position_log)


if __name__ == "__main__":
    run_and_generate_plots()
