# master_train_phase3.py
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from generate_expert_demos_phase3 import generate_demos

# BC model (64-64)
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

def train_bc(demo_path="data/demos_phase3.npz", epochs=30, lr=1e-3):
    data = np.load(demo_path, allow_pickle=True)
    demos = data["demos"]
    obs = np.vstack([d[0] for d in demos])
    acts = np.vstack([d[1] for d in demos])
    input_dim = obs.shape[1]
    output_dim = acts.shape[1]
    model = BCPolicy(input_dim, output_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(obs, dtype=torch.float32)
    Y = torch.tensor(acts, dtype=torch.float32)
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        print(f"BC Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/bc_phase3.pt")
    np.savez("models/bc_phase3_losses.npz", losses=np.array(losses))
    return "models/bc_phase3.pt"

def warmstart_ppo_with_bc(ppo_model, bc_path="models/bc_phase3.pt"):
    if not os.path.exists(bc_path):
        print("BC weights not found, skipping warmstart.")
        return 0
    bc_state = torch.load(bc_path)
    tmp_env = MultiRobotEnergyEnvPhase3()
    input_dim = tmp_env.observation_space.shape[0]
    output_dim = tmp_env.action_space.nvec.shape[0]
    bc = BCPolicy(input_dim, output_dim)
    bc.load_state_dict(bc_state)
    bc_params = list(bc.parameters())
    # find PPO policy net params
    try:
        ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
    except Exception as e:
        print("Cannot access PPO policy net; skipping warmstart.", e)
        return 0
    ppo_params = list(ppo_policy_net.parameters())
    copied = 0
    for ppo_p, bc_p in zip(ppo_params, bc_params):
        if ppo_p.shape == bc_p.shape:
            ppo_p.data.copy_(bc_p.data)
            copied += 1
    print(f"Copied {copied} tensors from BC -> PPO (shape-matching).")
    return copied

def train_ppo(total_timesteps=200_000):
    env_f = lambda: Monitor(MultiRobotEnergyEnvPhase3(enable_sharing=True, shaped_reward=True, debug=False))
    vec = DummyVecEnv([env_f])
    model = PPO("MlpPolicy", vec, verbose=1, tensorboard_log="./ppo_tb")
    warmstart_ppo_with_bc(model)
    model.learn(total_timesteps=total_timesteps)
    model.save("models/ppo_phase3")
    return "models/ppo_phase3.zip"

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("1) Generating expert demos (Phase 3 expert: cooperative)")
    demo_path = generate_demos(num_episodes=300, max_steps=400, save_path="data/demos_phase3.npz", enable_sharing=True)
    print("2) Training BC from demos")
    bc_path = train_bc(demo_path, epochs=30, lr=1e-3)
    print("3) Training PPO (warm-started)")
    ppo_path = train_ppo(total_timesteps=200_000)
    print("Done. Models saved in models/")
