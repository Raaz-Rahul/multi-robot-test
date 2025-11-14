# test_phase3_visualize_full.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
import os

MODEL = "models/ppo_full_phase3.zip"
if not os.path.exists(MODEL):
    MODEL = "models/ppo_multi_robot_coop.zip"  # fallback

print("Loading model:", MODEL)
model = PPO.load(MODEL)
env = MultiRobotEnergyEnvPhase3(debug=False)

NUM_EP = 5
all_rewards = []
all_steps = []

for ep in range(NUM_EP):
    obs, _ = env.reset()
    done = False
    step = 0
    positions = []
    batteries = []
    rewards = []
    events = []

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, trunc, info = env.step(action.tolist())

        states = next_obs.reshape(env.n_robots, 4)
        positions.append(states[:, :2].astype(int))
        batteries.append(states[:, 3])
        rewards.append(reward)

        if info.get("transfers"):
            events.append((step, "transfers", info["transfers"]))
        if info.get("deliveries"):
            events.append((step, "deliveries", info["deliveries"]))

        obs = next_obs
        step += 1

    total = sum(rewards)
    all_rewards.append(total)
    all_steps.append(step)
    print(f"\n=== Episode {ep+1} finished in {step} steps | reward {total:.2f} ===")
    if len(events) == 0:
        print("No transfers/deliveries recorded.")
    else:
        for e in events:
            print(e)

    # visualize the first episode with animation + plots
    if ep == 0:
        # animation
        fig, ax = plt.subplots(figsize=(6, 6))
        grid = env.grid_size
        def update(frame):
            ax.clear()
            ax.set_xlim(-1, grid)
            ax.set_ylim(-1, grid)
            ax.set_xticks(range(grid))
            ax.set_yticks(range(grid))
            ax.grid(True)
            pos = positions[frame]
            for i, (x, y) in enumerate(pos):
                ax.scatter(x, y, s=300)
                ax.text(x - 0.2, y + 0.2, f"R{i}", fontsize=12)
        ani = FuncAnimation(fig, update, frames=len(positions), interval=400)
        display(HTML(ani.to_jshtml()))
        plt.close()

        # battery plot
        plt.figure(figsize=(10,4))
        bats = np.array(batteries)
        for i in range(env.n_robots):
            plt.plot(bats[:, i], label=f"R{i}")
        plt.title("Battery trace (ep1)")
        plt.xlabel("Step")
        plt.ylabel("Battery")
        plt.legend()
        plt.grid()
        plt.show()

        # reward plot
        plt.figure(figsize=(10,3))
        plt.plot(rewards)
        plt.title("Per-step reward (ep1)")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

print("\nOverall avg reward:", np.mean(all_rewards), "avg steps:", np.mean(all_steps))
