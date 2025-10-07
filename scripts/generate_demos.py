"""
generate_demos.py
-----------------
Generates demonstration data using a simple heuristic policy
for the MultiRobotEnergyEnv base model.
"""

import os
import numpy as np
from tqdm import trange
from energy_env_multi import MultiRobotEnergyEnv


def greedy_action_towards(pos, dest):
    """Greedy move towards the destination (Manhattan heuristic)."""
    px, py = pos
    dx = dest[0] - px
    dy = dest[1] - py
    if abs(dx) >= abs(dy):
        if dx > 0:
            return 3  # East
        elif dx < 0:
            return 4  # West
    if dy > 0:
        return 1  # North
    elif dy < 0:
        return 2  # South
    return 0  # Stay


def run_episode(env, max_steps=200):
    """Run one heuristic episode and record transitions."""
    s_list, a_list, d_list = [], [], []
    obs, info = env.reset()
    dest = env.destination
    n = env.n

    for _ in range(max_steps):
        joint_action = []
        for i in range(n):
            x = int(obs[4 * i])
            y = int(obs[4 * i + 1])
            load = int(obs[4 * i + 2])
            if load > 0:
                a = greedy_action_towards((x, y), dest)
            else:
                a = 0
            joint_action.append(a)

        s_list.append(np.array(obs, dtype=np.float32))
        a_list.append(np.array(joint_action, dtype=np.int64))

        obs, reward, term, trunc, info = env.step(joint_action)
        done = term or trunc
        d_list.append(done)
        if done:
            break

    return np.array(s_list), np.array(a_list), np.array(d_list)


def main():
    os.makedirs("data", exist_ok=True)
    env = MultiRobotEnergyEnv(n_robots=3, grid_size=5, battery=30.0, debug=False)

    states, actions, dones = [], [], []

    for ep in trange(200, desc="Generating demos"):
        s, a, d = run_episode(env, max_steps=150)
        states.append(s)
        actions.append(a)
        dones.append(d)

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    dones = np.concatenate(dones, axis=0)

    np.savez_compressed("data/demos.npz", states=states, actions=actions, dones=dones)
    print("✅ Saved demonstration dataset to data/demos.npz")


if __name__ == "__main__":
    main()
