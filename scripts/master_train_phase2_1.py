# ==============================================================
# master_train_phase2.py
# Unified pipeline: Demos → BC → PPO (with Delivery Logging)
# ==============================================================

import os, sys, numpy as np, torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn

# --- Path fix (for Colab & VS Code) ---
if "__file__" in globals():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    repo_root = os.path.abspath("..")
sys.path.append(repo_root)

from energy_env_multi import MultiRobotEnergyEnv

# ==============================================================
# 1️⃣ Phase 1 - Generate Demonstrations
# ==============================================================
def generate_demos(n_episodes=200, max_steps=150, save_path="data/demos.npz"):
    print("\n🚀 Phase 1: Generating Demonstrations...")
    env = MultiRobotEnergyEnv(n_robots=3, grid_size=5, battery=50, debug=False)
    demos = []

    for ep in tqdm(range(n_episodes), desc="Generating demos"):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample().tolist()
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            demos.append((obs, action, reward, next_obs))
            obs = next_obs

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, demos=np.array(demos, dtype=object))  # ✅ fix
    print(f"✅ Demo generation complete — saved to {save_path}")
    return save_path


# ==============================================================
# 2️⃣ Phase 2 - Behavior Cloning (Supervised Learning)
# ==============================================================
def train_bc(demo_path="data/demos.npz", out_path="models/bc_policy.pt", epochs=20):
    print("\n🧠 Phase 2: Training Behavior Cloning (BC) model...")

    data = np.load(demo_path, allow_pickle=True)["demos"]
    obs = np.array([d[0] for d in data])
    acts = np.array([d[1] for d in data])

    model = nn.Sequential(
        nn.Linear(obs.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, len(acts[0]))
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred = model(torch.tensor(obs, dtype=torch.float32))
        loss = loss_fn(pred, torch.tensor(acts, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"✅ BC model saved to {out_path}")
    return out_path


# ==============================================================
# 3️⃣ Phase 3 - PPO Warm-start Training with Delivery Feedback
# ==============================================================
def train_ppo(bc_path="models/bc_policy.pt", total_timesteps=100000, out_path="models/ppo_multi_robot.zip"):
    print("\n🤖 Phase 3: Training PPO with BC warm-start...")

    env = make_vec_env(lambda: MultiRobotEnergyEnv(n_robots=3, grid_size=5, battery=50, debug=True), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1)

    if os.path.exists(bc_path):
        print("Loading BC weights into PPO model...")
        bc_state = torch.load(bc_path)
        model.policy.load_state_dict(bc_state, strict=False)

    # custom callback for delivery logging
    class DeliveryCallback:
        def __init__(self):
            self.step_count = 0
        def __call__(self, locals_, globals_):
            self.step_count += 1
            infos = locals_["infos"]
            for info in infos:
                if "violations" in info and info["violations"]:
                    print(f"⚠️ Constraint violation: {info['violations']}")
                if "delivery" in info:
                    print(f"🎯 Delivery detected (+{info['delivery']} reward)")
            return True

    callback = DeliveryCallback()

    model.learn(total_timesteps=total_timesteps, callback=callback.__call__)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"✅ PPO training complete — model saved to {out_path}")
    return out_path


# ==============================================================
# 🚀 MAIN PIPELINE
# ==============================================================
if __name__ == "__main__":
    demo_file = generate_demos()
    bc_model = train_bc(demo_file)
    ppo_model = train_ppo(bc_model)
    print("\n🎯 Phase 2 Master Training Completed Successfully!")
