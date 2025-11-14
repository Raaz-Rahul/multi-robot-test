# generate_expert_demos_phase3.py
import os
import numpy as np
from tqdm import trange
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

def manhattan_action(x, y, dx, dy):
    if x < dx:
        return 3
    if x > dx:
        return 4
    if y < dy:
        return 1
    if y > dy:
        return 2
    return 0

def generate_demos(num_episodes=300, max_steps=400, save_path="data/demos_phase3.npz", enable_sharing=True):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    env = MultiRobotEnergyEnvPhase3(enable_sharing=enable_sharing, shaped_reward=False, debug=False)
    demos = []
    for ep in trange(num_episodes, desc="Generating expert demos"):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            states = obs.reshape(env.n_robots, 5)  # [x,y,load,battery,Ehat]
            actions = [0] * env.n_robots
            for i, (x, y, load, battery, eh) in enumerate(states):
                if load == 0:
                    actions[i] = 0
                    continue
                # if battery low and helper nearby, request transfer
                if enable_sharing and battery < env.low_battery_threshold:
                    found = False
                    for j, (x2, y2, load2, bat2, eh2) in enumerate(states):
                        if j == i: continue
                        if load2 == 0 and bat2 > env.low_battery_threshold:
                            if abs(x - x2) + abs(y - y2) <= env.comm_range:
                                actions[i] = 5
                                found = True
                                break
                    if found:
                        continue
                # else move toward destination
                dx, dy = env.grid_size - 1, env.grid_size - 1
                actions[i] = manhattan_action(int(x), int(y), dx, dy)
            next_obs, reward, done, trunc, info = env.step(actions)
            demos.append((obs, np.array(actions, dtype=int), float(reward), next_obs))
            obs = next_obs
            step += 1
            # break when all delivered (expert ensures delivery)
            if info.get("deliveries_total", 0) == env.n_robots:
                break
    demos = np.array(demos, dtype=object)
    np.savez(save_path, demos=demos)
    print("Saved demos to:", save_path)
    return save_path

if __name__ == "__main__":
    generate_demos()
