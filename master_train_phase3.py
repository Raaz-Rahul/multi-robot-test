"""
Phase 3 Training Script – Cooperative Multi-Robot PPO with Imitation Warm Start
Author: Rahul 

This file:
1. Loads Phase-3 environment
2. Generates expert demonstrations
3. Performs Behaviour-Cloning warm start
4. Runs PPO fine-tuning
5. Saves final model to /models/
"""

import os
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from energy_env_phase3 import MultiRobotEnergyEnvPhase3


# ---------------------------------------------------------
#  HELPER: Save demos safely (no shape errors)
# ---------------------------------------------------------
def save_fixed_demos(path, demos):
    obs_list  = [d[0] for d in demos]
    act_list  = [d[1] for d in demos]
    rew_list  = [d[2] for d in demos]

    np.savez(
        path,
        observations=np.array(obs_list, dtype=object),
        actions=np.array(act_list, dtype=object),
        rewards=np.array(rew_list, dtype=object)
    )
    print(f"Expert demos saved safely → {path}")


# ---------------------------------------------------------
#  PHASE 3 — Generate expert demonstrations
# ---------------------------------------------------------
def generate_expert_demos(n_episodes=200, max_steps=200):
    env = MultiRobotEnergyEnvPhase3(debug=False)

    demos = []
    print("\nGenerating Expert Demos...")

    for _ in tqdm(range(n_episodes), desc="Expert Trajectories"):
        obs, _ = env.reset()
        for _ in range(max_steps):
            # Simple heuristic: robot moves toward destination greedily
            actions = []
            for r in range(env.n_robots):
                x, y, load, battery = env.robots[r]["x"], env.robots[r]["y"], env.robots[r]["load"], env.robots[r]["battery"]
                dx, dy = env.robots[r]["destination"]

                if x < dx:  a = 3  # east
                elif x > dx: a = 4  # west
                elif y < dy: a = 1  # north
                else: a = 0        # stay

                actions.append(a)

            next_obs, reward, done, trunc, info = env.step(actions)
            demos.append((obs, actions, reward))
            obs = next_obs
            if done:
                break

    os.makedirs("data", exist_ok=True)
    save_fixed_demos("data/phase3_demos.npz", demos)
    return "data/phase3_demos.npz"


# ---------------------------------------------------------
#  PHASE 3 — Behaviour Cloning warm-start
# ---------------------------------------------------------
def warm_start_with_bc(env, demo_file):
    print("\nBC Warm Start Started...")

    data = np.load(demo_file, allow_pickle=True)
    obs_arr = data["observations"]
    act_arr = data["actions"]

    model = PPO("MlpPolicy", env, verbose=0)
    pi = model.policy.mlp_extractor.policy_net  # get policy network

    optimizer = model.policy.optimizer

    import torch
    loss_fn = torch.nn.MSELoss()

    for epoch in range(3):
        epoch_loss = 0
        for obs, act in zip(obs_arr, act_arr):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor(act, dtype=torch.float32)

            pred = pi(obs_t)[0]
            loss = loss_fn(pred, act_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"BC Epoch {epoch+1} | Loss = {epoch_loss:.3f}")

    print("BC Warm Start Completed.\n")
    return model


# ---------------------------------------------------------
#  PHASE 3 — PPO fine-tuning
# ---------------------------------------------------------
def train_ppo_phase3():
    print("\n============================")
    print("Phase 3 PPO Training...")
    print("============================\n")

    env = MultiRobotEnergyEnvPhase3(debug=False)
    env = Monitor(env)

    demo_file = generate_expert_demos()

    # Warm Start
    bc_model = warm_start_with_bc(env, demo_file)

    # PPO Fine Tuning
    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99
    )

    # Load BC parameters into PPO
    ppo_model.policy.load_state_dict(bc_model.policy.state_dict())

    print(" Starting PPO Fine-Tuning...")
    ppo_model.learn(total_timesteps=15000)

    os.makedirs("models", exist_ok=True)
    ppo_model.save("models/ppo_cooperative_phase3")

    print("\n Phase 3 Training Complete!")
    print("Model saved → models/ppo_cooperative_phase3.zip")


if __name__ == "__main__":
    train_ppo_phase3()
