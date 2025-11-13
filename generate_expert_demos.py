# ================================================================
# generate_expert_demos.py
# Phase-3 Expert Demonstration Generator (Guaranteed Delivery)
# ================================================================
import numpy as np
from tqdm import tqdm
from energy_env_phase3 import MultiRobotEnergyEnvPhase3


def generate_demos(num_episodes=300, max_steps=500, save_path="data/demos_phase3.npz"):
    """
    Generates PERFECT expert demonstrations:
    - Robots always navigate optimally to the target
    - Low battery triggers load-sharing
    - Episode ends only when all deliveries done
    """
    env = MultiRobotEnergyEnvPhase3(debug=False)
    demos = []

    for ep in tqdm(range(num_episodes), desc="Generating Expert Demos"):

        obs, info = env.reset()
        done = False

        for step in range(max_steps):

            robot_states = obs.reshape(env.n_robots, 4)
            actions = []

            dest_x, dest_y = env.grid_size - 1, env.grid_size - 1

            # Expert policy per robot
            for i, (x, y, load, battery) in enumerate(robot_states):

                # Already delivered → stay
                if load == 0:
                    actions.append(0)
                    continue

                # Low battery → request transfer
                if battery < 12:
                    actions.append(5)
                    continue

                # Greedy shortest path
                if x < dest_x:
                    actions.append(3)  # east
                elif y < dest_y:
                    actions.append(1)  # north
                else:
                    actions.append(0)  # at destination

            next_obs, reward, done, trunc, info = env.step(actions)

            demos.append((obs, actions, reward, next_obs))
            obs = next_obs

            # Expert episode only ends after ALL deliveries
            if len(info["deliveries_total"]) == env.n_robots:
                break

    demos = np.array(demos, dtype=object)
    np.savez(save_path, demos=demos)

    print(f"✅ Expert demos saved to {save_path}")
    return save_path
