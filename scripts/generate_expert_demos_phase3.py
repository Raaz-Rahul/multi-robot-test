# generate_expert_demos.py
"""
Generate cooperative expert demonstrations for Phase 3.
Expert behavior:
 - If robot has load -> move along Manhattan shortest path towards destination
 - If robot has no load -> stay
 - Environment itself handles transfers when battery < threshold (comm_range/enabled)
Saves demos as an object-array .npz (demos list of tuples (obs, action, reward, next_obs))
"""
import numpy as np
import os
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from tqdm import trange

def expert_action_from_state(env, obs):
    # obs is flat array length 4*n : [x,y,load,battery,...]
    n = env.n_robots
    obs_rs = obs.reshape(n, 4)
    actions = []
    for i in range(n):
        x, y, load, bat = obs_rs[i]
        dest_x, dest_y = env.robots[i]["destination"]
        if load == 0:
            actions.append(0)  # stay if no load
            continue
        # Move along Manhattan shortest path: prefer x movement then y
        if x < dest_x:
            actions.append(3)  # east
        elif x > dest_x:
            actions.append(4)  # west
        elif y < dest_y:
            actions.append(1)  # north
        elif y > dest_y:
            actions.append(2)  # south
        else:
            actions.append(0)  # already at destination (should trigger delivery)
    return np.array(actions, dtype=int)

def generate_demos(num_episodes=500, max_steps=200, save_path="data/demos_expert_phase3.npz"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    env = MultiRobotEnergyEnvPhase3(debug=False)
    demos = []
    for ep in trange(num_episodes, desc="Generating expert demos"):
        obs, _ = env.reset()
        for t in range(max_steps):
            actions = expert_action_from_state(env, obs)
            next_obs, reward, done, trunc, info = env.step(actions.tolist())
            demos.append((obs, actions, reward, next_obs))
            obs = next_obs
            if done:
                break
    # Save as object array to avoid heterogeneous-shape errors
    np.savez(save_path, demos=np.array(demos, dtype=object))
    print("Saved demos:", save_path)
    return save_path

if __name__ == "__main__":
    generate_demos(num_episodes=300, max_steps=150, save_path="data/demos_expert_phase3.npz")
