# master_train_phase3.py
import os
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from generate_expert_demos import generate_demos

def supervised_warmstart(policy, demo_path, epochs=3, batch_size=128, lr=1e-3):
    """
    Try to perform supervised imitation updates on SB3 policy.
    Uses policy.get_distribution(obs) if available to compute log_prob of demo actions.
    If not available, will try a logits-based approach. If it fails, raise Exception.
    """
    data = np.load(demo_path, allow_pickle=True)
    obs_arr = data["observations"]
    act_arr = data["actions"]

    # flatten lists into arrays
    X = np.vstack([o for o in obs_arr]).astype(np.float32)              # (N, obs_dim)
    Y = np.vstack([a for a in act_arr]).astype(np.int64)               # (N, n_robots)

    device = next(policy.parameters()).device
    optimizer = policy.optimizer
    N = X.shape[0]

    print("BC warm-start: datapoints =", N)

    for ep in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0.0
        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            batch_obs = torch.tensor(X[idx], dtype=torch.float32, device=device)
            batch_actions = torch.tensor(Y[idx], dtype=torch.long, device=device)  # (B, n_robots)

            try:
                # try distribution-based negative log-likelihood if available
                dist = policy.get_distribution(batch_obs)
                # for MultiDiscrete, dist.log_prob expects shape matching; if not, try sum over dims
                logp = dist.log_prob(batch_actions)
                loss = -logp.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu().numpy())
            except Exception:
                # fallback: try to get logits from action_net (works for many SB3 versions)
                try:
                    features = policy.mlp_excriminator(batch_obs) if hasattr(policy, "mlp_discriminator") else None
                except Exception:
                    features = None
                try:
                    # use policy.action_net and policy.mlp_extractor where available
                    extractor = getattr(policy, "mlp_extractor", None)
                    if extractor is not None:
                        feats = extractor(batch_obs)
                    else:
                        feats = batch_obs
                    logits = policy.action_net(feats)
                    # split logits across robots
                    B, L = logits.shape
                    n_robots = batch_actions.shape[1]
                    head = L // n_robots
                    loss = torch.tensor(0.0, device=device)
                    for r in range(n_robots):
                        logit_r = logits[:, r*head:(r+1)*head]
                        act_r = batch_actions[:, r]
                        loss = loss + torch.nn.functional.cross_entropy(logit_r, act_r)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu().numpy())
                except Exception as e:
                    # if BC fails here, propagate so caller can fall back
                    raise RuntimeError("BC warm-start failed: " + str(e))
        print(f"BC epoch {ep+1}/{epochs} | loss={total_loss:.4f}")

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("1) Generating expert demos...")
    demo_path = generate_demos(num_episodes=200, max_steps=200, out_path="data/phase3_demos.npz")

    print("2) Creating environment and PPO model...")
    env = MultiRobotEnergyEnvPhase3(debug=False)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    model = PPO("MlpPolicy", vec_env, verbose=1)

    print("3) Attempting BC warm-start (best-effort)...")
    try:
        supervised_warmstart(model.policy, demo_path, epochs=3, batch_size=128, lr=1e-3)
        print("BC warm-start completed.")
    except Exception as e:
        print("BC warm-start failed (non-fatal). Continuing with PPO training.")
        print("BC error:", e)

    print("4) PPO fine-tuning...")
    model.learn(total_timesteps=50000)

    model.save("models/ppo_phase3")
    print("Saved PPO model to models/ppo_phase3.zip")

if __name__ == "__main__":
    main()
