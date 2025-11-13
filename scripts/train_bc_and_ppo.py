# train_bc_and_ppo.py
"""
Train BC from expert demos and warm-start PPO (Phase 3)
"""
import os
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

# ---------------- BC network (match SB3 default 64-64)
class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Generate or load demos
def maybe_generate_or_load(path="data/demos_expert_phase3.npz", force_generate=False):
    if not os.path.exists(path) or force_generate:
        from generate_expert_demos import generate_demos
        generate_demos(num_episodes=300, max_steps=150, save_path=path)
    return path

# ---------------- Train BC
def train_bc(demo_path="data/demos_expert_phase3.npz", epochs=20, lr=1e-3):
    data = np.load(demo_path, allow_pickle=True)
    demos = data["demos"]
    obs = np.vstack([d[0] for d in demos])
    actions = np.vstack([d[1] for d in demos])  # shape (N, n_robots)
    input_dim = obs.shape[1]
    output_dim = actions.shape[1]

    model = BCPolicy(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # predicting discrete actions as regression—works for warm-starting

    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bc_phase3.pt")
    print("Saved BC to models/bc_phase3.pt")
    return model

# ---------------- Warm-start PPO with BC (partial copy)
def warmstart_ppo_with_bc(ppo_model, bc_state_dict_path="models/bc_phase3.pt"):
    if not os.path.exists(bc_state_dict_path):
        print("BC weights not found, skipping warmstart.")
        return
    import torch
    bc_state = torch.load(bc_state_dict_path)
    # Build a temporary BC model to get ordered params
    env_tmp = MultiRobotEnergyEnvPhase3(debug=False)
    input_dim = env_tmp.observation_space.shape[0]
    output_dim = env_tmp.action_space.nvec.shape[0]
    bc_model = BCPolicy(input_dim, output_dim)
    bc_model.load_state_dict(bc_state)
    bc_params = list(bc_model.parameters())

    # Access PPO policy net (policy_net inside mlp_extractor)
    try:
        ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
        ppo_params = list(ppo_policy_net.parameters())
    except Exception:
        print("Couldn't locate PPO policy net parameters for warm-start. Skipping.")
        return

    # Copy matching-shaped params
    copied = 0
    for ppo_p, bc_p in zip(ppo_params, bc_params):
        if ppo_p.shape == bc_p.shape:
            with torch.no_grad():
                ppo_p.copy_(bc_p)
            copied += 1
    print(f"Warm-started PPO by copying {copied} parameter tensors (matching shapes).")

# ---------------- Train PPO
def train_ppo(total_timesteps=250_000, bc_path="models/bc_phase3.pt"):
    env = DummyVecEnv([lambda: MultiRobotEnergyEnvPhase3(debug=False)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_phase3_tb")
    # Warmstart (partial)
    warmstart_ppo_with_bc(model, bc_state_dict_path=bc_path)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_multi_robot_coop")
    print("Saved PPO to models/ppo_multi_robot_coop.zip")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    demo_path = maybe_generate_or_load(path="data/demos_expert_phase3.npz", force_generate=False)
    train_bc(demo_path, epochs=30, lr=1e-3)
    train_ppo(total_timesteps=120_000, bc_path="models/bc_phase3.pt")
