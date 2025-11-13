import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from IPython.display import HTML, display

# ------------------------------------------
# Load environment + trained PPO model
# ------------------------------------------
env = MultiRobotEnergyEnvPhase3(debug=False)
model = PPO.load("models/ppo_multi_robot_coop.zip")

# ------------------------------------------
# Run evaluation episode
# ------------------------------------------
obs, info = env.reset()
done = False

positions = []    # robot (x,y)
batteries = []    # battery levels
rewards = []      # per-step reward
events = []       # transfer + delivery logs

step = 0
max_steps = 200

while not done and step < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    next_obs, reward, done, trunc, info = env.step(action.tolist())

    robot_states = next_obs.reshape(env.n_robots, 4)
    pos = robot_states[:, :2].astype(int)
    batt = robot_states[:, 3]

    positions.append(pos)
    batteries.append(batt)
    rewards.append(reward)

    if "deliveries" in info and info["deliveries"]:
        events.append(f"Step {step}: Delivered by robots {info['deliveries']}")
    if "transfers" in info and info["transfers"]:
        events.append(f"Step {step}: Transfer {info['transfers']}")

    obs = next_obs
    step += 1

print("\n=== Events During Episode ===")
if len(events) == 0:
    print("(No deliveries or transfers detected)")
else:
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
    ax.grid(True)

    pos = positions[frame]
    for i, (x, y) in enumerate(pos):
        ax.scatter(x, y, s=300, marker="o")
        ax.text(x - 0.2, y + 0.2, f"R{i}", fontsize=12)

ani = FuncAnimation(fig, update, frames=len(positions), interval=600)

# FORCE DISPLAY IN COLAB:
print("\nDisplaying robot movement animation below...")
display(HTML(ani.to_jshtml()))

# ------------------------------------------
# Visualization 2: Battery Plot
# ------------------------------------------
plt.figure(figsize=(10, 5))
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
# Visualization 3: Reward Plot
# ------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(rewards, color="purple")
plt.title("Reward Per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid()
plt.show()
