# demo_run.py
# Quick demo: shows action vector, state vector, violations, reward, deliveries

import numpy as np
from env import MultiRobotEnv
from copy import deepcopy
import pprint
pp = pprint.PrettyPrinter(indent=2)

env = MultiRobotEnv(grid_size=(8,10), n_robots=3, init_battery=100.0, share_radius=3, max_steps=50)
obs = env.reset()
print("Initial state vector length:", obs['state'].shape[0])
env.render()
print()

# Helper to build action: list of per-robot blocks [move, offload, partner, meeting_idx, w_idx]
def make_action(moves, offloads=None, partners=None, meetings=None, w_idxs=None):
    n = env.n
    if offloads is None: offloads = [0]*n
    if partners is None: partners = [0]*n
    if meetings is None: meetings = [env.node_to_idx[env.positions[i]] for i in range(n)]
    if w_idxs is None: w_idxs = [0]*n
    act = []
    for i in range(n):
        act.extend([int(moves[i]), int(offloads[i]), int(partners[i]), int(meetings[i]), int(w_idxs[i])])
    return np.array(act, dtype=int)

# Scenario 1: Move robots down (2) and right (4) in alternating steps to demonstrate movement
actions = [
    make_action([2,2,2]),  # all move down
    make_action([4,4,4]),  # all move right
    make_action([4,0,0]),  # robot0 right, others stay
    make_action([1,1,1]),  # all move up
]

for step, act in enumerate(actions):
    old_pos = deepcopy(env.positions)
    old_loads = deepcopy(env.loads)
    obs, reward, done, info = env.step(act)
    print(f"\n--- STEP {step} ---")
    print("action_vector:", info['action_vector'].tolist())
    print("state_vector (len):", info['state_vector'].shape[0])
    # movement summary
    for i in range(env.n):
        if old_pos[i] != env.positions[i]:
            print(f"Robot {i} moved {old_pos[i]} -> {env.positions[i]}")
        else:
            print(f"Robot {i} stayed at {env.positions[i]}")
    # loads and deliveries
    for i in range(env.n):
        if env.loads[i] != old_loads[i]:
            print(f"Robot {i} load changed {old_loads[i]} -> {env.loads[i]}")
    if any(info['delivered'].values()):
        print("Delivered this step:", {k:v for k,v in info['delivered'].items() if v})
    # violations
    print("violations:")
    pp.pprint(info['violations'])
    print("sharing_events:", info.get('sharing_events', []))
    print("collisions:", info.get('collisions', []))
    print("energy:", info['energy'], "reward:", info['reward'])
    env.render()
    if done:
        print("Env done.")
        break

print("\nEnd demo.")
