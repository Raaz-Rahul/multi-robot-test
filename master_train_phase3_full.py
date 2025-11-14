# master_train_phase3_full.py
"""
Full Phase-3 master train:
 - generate expert demos (guaranteed delivery)
 - train Behavior Cloning (BC) (64-64 MLP)
 - warm-start PPO (copy matching shapes)
 - train PPO with recommended hyperparams
"""
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from energy_env_phase3 import MultiRobotEnergyEnvPhase3

# BC network (64-64)
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

# ---------------- produce/load demos ----------------
def ensure_demos(path="data/demos_full_phase3.npz", force=False):
    if not os.path.exists(path) or force:
        from generate_expert_demos_full import generate_demos
        generate_demos(num_episodes=400, max_steps=400, save_path=path)
    return path

# ---------------- train BC ----------------
def train_bc(demo_path="data/demos_full_phase3.npz", epochs=30, lr=1e-3):
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

    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        print(f"BC Epoch {ep+1}/{epochs} loss={loss.item():.6f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/bc_full_phase3.pt")
    print("Saved BC -> models/bc_full_phase3.pt")
    return "models/bc_full_phase3.pt"

# ---------------- warm-start PPO (shape-safe copying) ----------------
def warmstart_ppo_with_bc(ppo_model, bc_state_path="models/bc_full_phase3.pt"):
    if not os.path.exists(bc_state_path):
        print("No BC weights found; skipping warmstart")
        return
    bc_state = torch.load(bc_state_path)
    tmp_env = MultiRobotEnergyEnvPhase3()
    input_dim = tmp_env.observation_space.shape[0]
    output_dim = tmp_env.action_space.nvec.shape[0]
    bc_model = BCPolicy(input_dim, output_dim)
    bc_model.load_state_dict(bc_state)
    bc_params = list(bc_model.parameters())

    # try to access ppo policy net params
    try:
        ppo_policy_net = ppo_model.policy.mlp_extractor.policy_net
    except Exception as e:
        print("Couldn't find ppo.policy.mlp_extractor.policy_net; skipping warmstart", e)
        return
    ppo_params = list(ppo_policy_net.parameters())

    copied = 0
    for ppo_p, bc_p in zip(ppo_params, bc_params):
        if ppo_p.shape == bc_p.shape:
            ppo_p.data.copy_(bc_p.data)
            copied += 1
    print(f"Copied {copied} parameter tensors from BC -> PPO (shape-matching)")

# ---------------- train PPO ----------------
def train_ppo(total_timesteps=300_000, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    env = DummyVecEnv([lambda: MultiRobotEnergyEnvPhase3(**env_kwargs)])
    model = PPO("MlpPolicy", env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                ent_coef=0.01,
                vf_coef=0.5,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log="./ppo_phase3_tb")
    # warmstart
    warmstart_ppo_with_bc(model)
    model.learn(total_timesteps=total_timesteps)
    Path("models").mkdir(exist_ok=True)
    model.save("models/ppo_full_phase3")
    print("Saved PPO -> models/ppo_full_phase3.zip")
    return "models/ppo_full_phase3.zip"

# ---------------- main ----------------
if __name__ == "__main__":
    print("Phase 3 Full Hybrid training start.")
    os.makedirs("data", exist_ok=True)
    demo_path = ensure_demos()
    bc_path = train_bc(demo_path, epochs=30, lr=1e-3)
    ppo_path = train_ppo(total_timesteps=300_000)
    print("Training finished. BC:", bc_path, "PPO:", ppo_path)
