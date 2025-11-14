# generate_expert_demos_full.py
"""
Deterministic expert generator for Full Hybrid behaviour:
 - Manhattan shortest-path navigation to each robot's destination
 - Cooperative transfer when requester battery < threshold AND helper is nearby
 - Ensures episodes end only after all deliveries or max_steps
 - Saves demos as object-array .npz for BC training
"""
import os
import numpy as np
from tqdm import trange
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

def manhattan_next_action(x, y, dest_x, dest_y):
    # prefer x movement then y (deterministic)
    if x < dest_x:
        return 3  # east
    if x > dest_x:
        return 4  # west
    if y < dest_y:
        return 1  # north
    if y > dest_y:
        return 2  # south
    return 0  # stay (at destination)

def generate_demos(num_episodes=400, max_steps=400, save_path="data/demos_full_phase3.npz",
                   low_battery_transfer_threshold=12):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    env = MultiRobotEnergyEnvPhase3(debug=False)

    demos = []

    for ep in trange(num_episodes, desc="Generating expert demos"):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            states = obs.reshape(env.n_robots, 4)  # [x,y,load,battery]
            actions = [0] * env.n_robots

            # 1) Plan movement for robots that have load
            for i, (x, y, load, bat) in enumerate(states):
                if load == 0:
                    actions[i] = 0
                else:
                    # If battery low -> request transfer (action 5)
                    if bat < low_battery_transfer_threshold:
                        # but only request if any available helper nearby (heuristic: check manhattan dist <= comm_range)
                        found_helper = False
                        for j, (x2, y2, load2, bat2) in enumerate(states):
                            if j == i: continue
                            if load2 == 0 and bat2 > low_battery_transfer_threshold:
                                if abs(x - x2) + abs(y - y2) <= env.comm_range:
                                    found_helper = True
                                    break
                        if found_helper:
                            actions[i] = 5
                            continue
                        # if no helper nearby, continue moving towards goal to try to reach helper or destination
                    # otherwise follow Manhattan shortest path
                    dest_x, dest_y = env.grid_size - 1, env.grid_size - 1
                    actions[i] = manhattan_next_action(int(x), int(y), dest_x, dest_y)

            # 2) Execute step
            next_obs, reward, done, trunc, info = env.step(actions)
            demos.append((obs, np.array(actions, dtype=int), float(reward), next_obs))
            obs = next_obs
            steps += 1

            # If environment signals all delivered, break (expert satisfied)
            if info.get("deliveries_total", 0) == env.n_robots:
                break

    demos = np.array(demos, dtype=object)
    np.savez(save_path, demos=demos)
    print("Saved demos to:", save_path)
    return save_path

if __name__ == "__main__":
    generate_demos()
