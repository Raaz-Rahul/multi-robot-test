# master_train_phase3.py
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from generate_expert_demos import generate_demos

def train_bc_on_policy(policy, demo_path, epochs=3, lr=1e-3, batch_size=64):
    """
    Perform supervised updates on PPO policy network using demo data.
    policy: model.policy (a Stable-Baselines policy object)
    """
    data = np.load(demo_path, allow_pickle=True)
    obs_arr = data["observations"]
    act_arr = data["actions"]

    # prepare all training pairs
    X = np.vstack([o for o in obs_arr])
    Y = np.vstack([a for a in act_arr])  # shape (N, n_robots)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.long)  # indices

    optimizer = policy.optimizer
    device = next(policy.parameters()).device

    N = X_t.shape[0]
    idxs = np.arange(N)

    # policy.forward for SB3 MlpPolicy produces distribution; we will get logits from policy.net
    for ep in range(epochs):
        np.random.shuffle(idxs)
        total_loss = 0.0
        for start in range(0, N, batch_size):
            batch_idx = idxs[start:start+batch_size]
            batch_obs = X_t[batch_idx].to(device)
            batch_actions = Y_t[batch_idx].to(device)  # shape (B, n_robots)
            # forward through policy to get action logits for discrete multi-action
            # SB3 policy has action_net producing distribution parameters; we use policy._predict() indirectly
            # Simpler: get features from policy.mlp_extractor and then use policy.action_net
            with torch.enable_grad():
                features = policy.mlps_extractor(batch_obs) if hasattr(policy, "mlps_extractor") else None
                # different SB3 versions: use policy.mlp_extractor.features if available
                # robust approach: call policy._get_flat_acts? We'll instead call policy.forward's net
                # Use policy._predict to obtain ndarray actions - but we need logits; instead compute MSE on continuous prox:
                # Approach: use policy.action_net(features) if available. We'll try common SB3 attr names.
                # Fallback: perform gradient-free imitation by minimizing difference between chosen action indices and predicted continuous output.
                try:
                    # For MlpPolicy, policy.action_net exists and returns logits for Discrete
                    logits = policy.action_net(policy.mlp_extractor(batch_obs))
                except Exception:
                    # Fallback: run policy.forward to get distribution and compute cross-entropy using dist.log_prob
                    dist = policy.get_distribution(batch_obs)
                    loss = -dist.log_prob(batch_actions).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu().numpy())
                    continue

                # We have logits shaped (B, sum(action_dims)) or (B, n_actions) depending on implementation.
                # For MultiDiscrete, SB3 often flattens multiple categorical heads; fallback: compute cross-entropy per robot.
                # We'll attempt to split logits into n_robots heads by equal partition.
                B, L = logits.shape
                n_robots = batch_actions.shape[1]
                head_dim = L // n_robots
                loss = torch.tensor(0.0, device=device)
                for r in range(n_robots):
                    logit_r = logits[:, r*head_dim:(r+1)*head_dim]
                    act_r = batch_actions[:, r]
                    loss += torch.nn.functional.cross_entropy(logit_r, act_r)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().numpy())
        print(f"BC epoch {ep+1}/{epochs} | loss={total_loss:.4f}")

def main():
    os.makedirs("models", exist_ok=True)
    print("1) Generating expert demos...")
    demo_path = generate_demos(num_episodes=250, max_steps=200, out_path="data/phase3_demos.npz")
    print("2) Creating environment and PPO model...")
    env = MultiRobotEnergyEnvPhase3(debug=False)
    vec = DummyVecEnv([lambda: Monitor(env)])
    model = PPO("MlpPolicy", vec, verbose=1)

    print("3) Warm-starting policy with BC (supervised updates)...")
    # run a few supervised updates on model.policy using demos
    train_bc_on_policy(model.policy, demo_path, epochs=5, lr=1e-3, batch_size=128)

    print("4) PPO fine-tuning...")
    model.learn(total_timesteps=50000)

    model.save("models/ppo_phase3")
    print("Model saved to models/ppo_phase3.zip")

if __name__ == "__main__":
    main()
