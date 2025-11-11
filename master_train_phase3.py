"""
master_train_phase3.py
Phase 3: Cooperative Multi-Robot PPO training with load-sharing and BC warm start.
Author: Rahul Kapardar
"""

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

# -----------------------------------------------------------
# Setup directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------------------------------------
# 1️⃣ Generate demonstrations (random actions for BC pretraining)
def generate_demos(num_episodes=200, max_steps=100, save_path="data/demos_phase3.npz"):
    print("\n🚀 Phase 1: Generating Cooperative Demonstrations...")
    env = MultiRobotEnergyEnvPhase3(debug=False)
    demos = []

    for ep in tqdm(range(num_episodes), desc="Generating demos"):
        obs, _ = env.reset()
        for t in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, done, trunc, info = env.step(action)
            demos.append((obs, action, reward, next_obs))
            obs = next_obs
            if done:
                break

    # ✅ Fix: Save using allow_pickle=True
    np.savez(save_path, demos=np.array(demos, dtype=object))
    print(f"✅ Demo generation complete — saved to {save_path}")
    return save_path


# -----------------------------------------------------------
# 2️⃣ Behavior Cloning (BC) Network
class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )


    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------
def train_bc(demo_path="data/demos_phase3.npz", num_epochs=20, lr=1e-3):
    print("\n🧠 Phase 2: Training Behavior Cloning (BC) model...")
    data = np.load(demo_path, allow_pickle=True)
    demos = data["demos"]

    obs = np.vstack([d[0] for d in demos])
    actions = np.vstack([d[1] for d in demos])

    input_dim = obs.shape[1]
    output_dim = actions.shape[1]
    model = BCPolicy(input_dim, output_dim)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(torch.tensor(obs, dtype=torch.float32))
        loss = loss_fn(pred, torch.tensor(actions, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.5f}")

    torch.save(model.state_dict(), "models/bc_phase3.pt")
    print("✅ BC model saved to models/bc_phase3.pt")
    return model

# -----------------------------------------------------------
# 3️⃣ PPO Training with Cooperative Environment and Warm Start
def train_ppo_with_warmstart():
    print("\n🤖 Phase 3: Training PPO with Cooperative Load Sharing...")

    env = MultiRobotEnergyEnvPhase3(debug=False)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="ppo_phase3_tensorboard")

    # Load pretrained BC weights (optional)
    bc_path = "models/bc_phase3.pt"
    if os.path.exists(bc_path):
        print("Loading BC weights into PPO model...")
        bc_model = BCPolicy(env.observation_space.shape[0], env.action_space.shape[0])
        bc_model.load_state_dict(torch.load(bc_path))
        bc_model.eval()
        ppo_policy = model.policy.mlp_extractor.policy_net
        with torch.no_grad():
            for ppo_param, bc_param in zip(ppo_policy.parameters(), bc_model.parameters()):
                ppo_param.copy_(bc_param)
    else:
        print("⚠️ BC weights not found, training PPO from scratch.")

    # Train PPO
    model.learn(total_timesteps=120_000)
    model.save("models/ppo_multi_robot_coop.zip")
    print("✅ PPO training complete — model saved to models/ppo_multi_robot_coop.zip")

# -----------------------------------------------------------
if __name__ == "__main__":
    demo_file = generate_demos()
    train_bc(demo_file)
    train_ppo_with_warmstart()
    print("\n🎯 Phase 3 Cooperative Training Completed Successfully!")


