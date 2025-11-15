# generate_expert_demos.py
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

def generate_demos(num_episodes=300, max_steps=200, out_path="data/phase3_demos.npz"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    env = MultiRobotEnergyEnvPhase3(debug=False)
    demos = []
    for ep in trange(num_episodes, desc="Generating demos"):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            states = obs.reshape(env.n_robots, 5)
            actions = []
            for i, s in enumerate(states):
                x, y, load, battery, ehat = s
                if load == 0:
                    actions.append(0)
                    continue
                # if low battery and helper idle nearby: request transfer
                if battery < env.low_battery_threshold:
                    found = False
                    for j, sj in enumerate(states):
                        if j == i: continue
                        if sj[2] == 0 and sj[3] > env.low_battery_threshold:
                            if abs(x - sj[0]) + abs(y - sj[1]) <= env.comm_range:
                                actions.append(5)  # request transfer
                                found = True
                                break
                    if found:
                        continue
                # else move towards destination
                dx, dy = env.grid_size - 1, env.grid_size - 1
                actions.append(manhattan_action(int(x), int(y), dx, dy))
            next_obs, reward, done, _, info = env.step(actions)
            demos.append((obs, np.array(actions, dtype=int), float(reward)))
            obs = next_obs
            steps += 1
    # save safely as object arrays
    obs_list = [d[0] for d in demos]
    act_list = [d[1] for d in demos]
    rew_list = [d[2] for d in demos]
    np.savez(out_path,
             observations=np.array(obs_list, dtype=object),
             actions=np.array(act_list, dtype=object),
             rewards=np.array(rew_list, dtype=object))
    print("Saved demos to:", out_path)
    return out_path

if __name__ == "__main__":
    generate_demos()
