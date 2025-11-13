import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

# ------------------------------------------
# Load environment and trained PPO model
# ------------------------------------------
env = MultiRobotEnergyEnvPhase3(debug=False)
model = PPO.load("models/ppo_multi_robot_coop.zip")

# ------------------------------------------
# Run one evaluation episode
# ------------------------------------------
obs, info = env.reset()
done = False

positions = []    # Tracks robot positions per step
batteries = []    # Tracks battery values per step
rewards = []      # Tracks reward per step
events = []       # Delivery & transfer log

step = 0

while not done and step < 200:
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, trunc, info = env.step(action.tolist())

    robot_states = next_obs.reshape(env.n_robots, 4)

    # Save positions and batteries
    pos = robot_states[:, :2].astype(int)
    batt = robot_states[:, 3]

    positions.append(pos)
    batteries.append(batt)
    rewards.append(reward)

    # Log events
    if "deliveries" in info and len(info["deliveries"]) > 0:
        events.append(f"Step {step}: Deliveries by robots {info['deliveries']}")
    if "transfers" in info and len(info["transfers"]) > 0:
        events.append(f"Step {step}: Load transfer {info['transfers']}")

    obs = next_obs
    step += 1

print("\n=== Events During Episode ===")
for e in events:
    print(e)

# ------------------------------------------
# Visualization 1: Robot Movement Animation
# ------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))

grid = env.grid_size

def update(frame):
    ax.clear()
    ax.set_title(f"Robot Movement (Step {frame})")
    ax.set_xlim(-1, grid)
    ax.set_ylim(-1, grid)
    ax.set_xticks(range(grid))
    ax.set_yticks(range(grid))

    pos = positions[frame]
    for i, (x, y) in enumerate(pos):
        ax.scatter(x, y, s=400, label=f"R{i}")
        ax.text(x - 0.2, y + 0.2, f"R{i}", fontsize=12)

ani = FuncAnimation(fig, update, frames=len(positions), interval=500)
plt.close()  # Prevent double display

print("\nRobot movement animation ready.")
from IPython.display import HTML
HTML(ani.to_jshtml())

# ------------------------------------------
# Visualization 2: Battery Consumption Plot
# ------------------------------------------
plt.figure(figsize=(8, 5))
batteries_array = np.array(batteries)

for i in range(env.n_robots):
    plt.plot(batteries_array[:, i], label=f"Robot {i}")

plt.title("Battery Consumption per Robot")
plt.xlabel("Step")
plt.ylabel("Battery Level")
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------
# Visualization 3: Reward per Step
# ------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(rewards, color="purple")
plt.title("Reward Per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.show()
