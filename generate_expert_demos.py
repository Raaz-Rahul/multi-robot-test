# ================================================================
# generate_expert_demos.py
# Phase-3 Expert Demonstration Generator (Cooperative Load-Sharing)
# ================================================================
import numpy as np
from tqdm import tqdm
from energy_env_phase3 import MultiRobotEnergyEnvPhase3


def generate_demos(num_episodes=150, max_steps=200, save_path="data/demos_phase3.npz"):
    """
    Generates expert demonstrations using simple cooperative rules:
    - Robots move toward destination
    - Low-battery robots transfer load to nearest available robot
    """

    env = MultiRobotEnergyEnvPhase3(debug=False)

    demos = []

    for ep in tqdm(range(num_episodes), desc="Generating Expert Demos"):

        obs, info = env.reset()
        done = False

        for step in range(max_steps):
            # --------------------------------------
            # EXPERT POLICY (Handcrafted)
            # --------------------------------------
            actions = []
            robot_states = obs.reshape(env.num_robots, 4)

            for i, (x, y, load, battery) in enumerate(robot_states):

                # Robot already delivered? stay still
                if load == 0:
                    actions.append(0)  # stay
                    continue

                # If battery low → allow transfer action
                if battery < 10:
                    actions.append(5)  # ACTION 5 → Request load transfer
                    continue

                # Otherwise → move toward destination (grid-1, grid-1)
                dest_x, dest_y = env.grid_size - 1, env.grid_size - 1

                if x < dest_x:
                    actions.append(3)  # move east
                elif y < dest_y:
                    actions.append(1)  # move north
                else:
                    actions.append(0)  # at destination → stay

            # Step environment
            next_obs, reward, done, trunc, info = env.step(actions)

            # Save transition
            demos.append((obs, actions, reward, next_obs))

            obs = next_obs
            if done:
                break

    # Convert to object array so numpy can save properly
    demos = np.array(demos, dtype=object)
    np.savez(save_path, demos=demos)

    print(f"✅ Expert demos saved to {save_path}")
    return save_path
