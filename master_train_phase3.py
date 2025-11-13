# master_train_phase3.py
"""
Phase 3 Training Script:
 - Loads/Generates Cooperative Expert Demos
 - Trains Behavior Cloning (BC)
 - Warm-starts PPO with BC weights (shape-safe)
 - Trains PPO on cooperative load-sharing environment
"""

import numpy as np
import os
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from energy_env_phase3 import MultiRobotEnergyEnvPhase3


# ------------------ BC Network (Match SB3: 64-64 MLP) ------------------
class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# ------------------ Load or Generate Demos ------------------
def load_or_generate_demos(path="data/demos_expert_phase3.npz"):
    if not os.path.exists(path):
        from generate_expert_demos import generate_demos
        print("Generating expert demos...")
        generate_demos(save_path=path)
    return path


# ------------------ Train BC ------------------
def train_bc(demo_path="data/demos_expert_phase3.npz", epochs=25):
    data = np.load(demo_path, allow_pickle=True)
    demos = data["demos"]

    obs = np.vstack([d[0] for d in demos])
    actions = np.vstack([d[1] for d in demos])

    input_dim = obs.shape[1]
    output_dim = actions.shape[1]

    model = BCPolicy(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    print("\n🧠 Training Behavior Cloning Model...")
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {ep+1}/{epochs} | BC Loss = {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bc_phase3.pt")
    print("\n✅ BC model saved to models/bc_phase3.pt")

    return model


# ------------------ Warm-Start PPO using BC ------------------
def warmstart_ppo_with_bc(ppo_model, bc_path="models/bc_phase3.pt"):
    if not os.path.exists(bc_path):
        print("⚠ BC weights not found. Skipping warmstart.")
        return

    bc_state = torch.load(bc_path)

    # Build temp BC model to match parameters
    tmp_env = MultiRobotEnergyEnvPhase3()
    input_dim = tmp_env.observation_space.shape[0]
    output_dim = tmp_env.action_space.nvec.shape[0]

    bc_model = BCPolicy(input_dim, output_dim)
    bc_model.load_state_dict(bc_state)

    bc_params = list(bc_model.parameters())

    try:
        ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
        ppo_params = list(ppo_policy_net.parameters())
    except:
        print("Could not access PPO policy params.")
        return

    copied = 0
    for ppo_p, bc_p in zip(ppo_params, bc_params):
        if ppo_p.shape == bc_p.shape:
            ppo_p.data.copy_(bc_p.data)
            copied += 1

    print(f"\n🔄 Warm-started PPO: {copied} layers copied from BC.")


# ------------------ Train PPO ------------------
def train_ppo(total_timesteps=150_000):
    print("\n🤖 Initializing PPO...")
    env = DummyVecEnv([lambda: MultiRobotEnergyEnvPhase3(debug=False)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_phase3_tb")

    warmstart_ppo_with_bc(model)

    print("\n🚀 Training PPO...")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_multi_robot_coop")
    print("\n🎉 PPO model saved: models/ppo_multi_robot_coop.zip")


# ------------------ Main ------------------
if __name__ == "__main__":
    print("\n🚀 Phase 3 Training Started...")
    demo_file = load_or_generate_demos()
    train_bc(demo_file)
    train_ppo()
    print("\n🎯 Phase 3 Training Completed Successfully!\n")
