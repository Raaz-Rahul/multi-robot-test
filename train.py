# train.py
# BC warm-start followed by PPO (single-env on-policy demo)

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from env import MultiRobotEnv
from policy import MaskedPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# Config
ENV_CONFIG = dict(grid_size=(8,8), n_robots=2, init_battery=50.0, a=1.0, b=0.1, share_radius=3, beta=0.8, max_load_per_robot=10, max_steps=200)
BC_EPOCHS = 6
PPO_EPOCHS = 60
ROLLOUT_STEPS = 128
MINI_BATCHES = 4
PPO_CLIP = 0.1
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
LR = 3e-4

# -------------------------
# Utilities
# -------------------------
def flatten_obs(obs):
    return torch.tensor(obs['state'], dtype=torch.float32, device=DEVICE).unsqueeze(0)

def make_action_from_labels(labels, env):
    act = []
    for i in range(env.n):
        mv = 0; off = 0; partner = 0; meeting_idx = env.node_to_idx[env.positions[i]]; w_idx = 0
        if i in labels:
            off = labels[i]['offload']
            partner = labels[i]['partner']
            meeting_idx = env.node_to_idx.get(labels[i]['meeting_node'], meeting_idx)
            w = labels[i]['w']
            if w in env.w_bins:
                w_idx = env.w_bins.index(w)
        act.extend([mv, off, partner, meeting_idx, w_idx])
    return np.array(act, dtype=np.int64)

def generate_demo_dataset(env, n_episodes=30):
    X = []; Y = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False; steps = 0
        while not done and steps < 50:
            labels = env.generate_demonstration()
            action = make_action_from_labels(labels, env)
            X.append(obs['state'].copy()); Y.append(action.copy())
            obs, _, done, _ = env.step(action)
            steps += 1
    X = np.stack(X) if X else np.zeros((0, env._get_obs()['state'].shape[0]))
    Y = np.stack(Y) if Y else np.zeros((0, env.n * len(env.per_robot_dims)), dtype=np.int64)
    return X, Y

def bc_train(policy, X, Y, optimizer, epochs=6, batch_size=64):
    policy.train()
    criterion = torch.nn.CrossEntropyLoss()
    N = X.shape[0]
    for ep in range(epochs):
        idx = np.arange(N); np.random.shuffle(idx)
        losses = []
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start+batch_size]
            bx = torch.tensor(X[batch_idx], dtype=torch.float32, device=DEVICE)
            by = torch.tensor(Y[batch_idx], dtype=torch.int64, device=DEVICE)
            logits_per_robot, _ = policy(bx)
            loss = 0.0
            for ri in range(policy.n):
                logits = logits_per_robot[ri]
                startcol = 0
                for hid, size in enumerate(policy.per_robot_dims):
                    seg = logits[:, startcol:startcol+size]
                    target = by[:, ri*len(policy.per_robot_dims) + hid]
                    loss = loss + criterion(seg, target)
                    startcol += size
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        print(f"BC epoch {ep+1}/{epochs} avg loss {np.mean(losses):.4f}")

def compute_gae(rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    T = len(rewards)
    values = np.append(values, 0.0)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = masks[t]
        delta = rewards[t] + gamma * values[t+1] * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns

# -------------------------
# Main
# -------------------------
def main():
    env = MultiRobotEnv(**ENV_CONFIG)
    obs0 = env.reset()
    obs_dim = obs0['state'].shape[0]

    policy = MaskedPolicy(obs_dim, env.n, env.per_robot_dims).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Demos and BC
    X_demo, Y_demo = generate_demo_dataset(env, n_episodes=40)
    print("Demo shapes:", X_demo.shape, Y_demo.shape)
    if X_demo.shape[0] > 0:
        bc_train(policy, X_demo, Y_demo, optimizer, epochs=BC_EPOCHS)
        torch.save(policy.state_dict(), os.path.join(MODEL_DIR, "policy_bc.pt"))

    # PPO training (single-env simplified)
    for epoch in range(PPO_EPOCHS):
        obs = env.reset()
        states = []; actions = []; old_logps = []; rewards = []; masks = []; values = []
        for step in range(ROLLOUT_STEPS):
            bx = flatten_obs(obs)
            logits_per_robot, value = policy(bx)
            acts, lp = policy.sample_actions([l for l in logits_per_robot], [obs['masks']], device=DEVICE)
            act = acts[0].cpu().numpy()
            lp_val = lp[0].item()
            obs2, r, done, info = env.step(act)
            states.append(obs['state'])
            actions.append(act)
            old_logps.append(lp_val)
            rewards.append(r)
            masks.append(0.0 if done else 1.0)
            values.append(value.detach().cpu().item())
            obs = obs2
            if done:
                obs = env.reset()
        advantages, returns = compute_gae(rewards, masks, np.array(values))
        # to tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(np.array(actions), dtype=torch.int64, device=DEVICE)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        old_logps_t = torch.tensor(np.array(old_logps), dtype=torch.float32, device=DEVICE)

        N = states_t.shape[0]
        inds = np.arange(N)
        for _ in range(4):
            np.random.shuffle(inds)
            for start in range(0, N, max(1, N // MINI_BATCHES)):
                mb_idx = inds[start:start + max(1, N // MINI_BATCHES)]
                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                # approximate masks for this batch
                mb_masks = [env._get_obs()['masks'] for __ in mb_idx]
                new_logps, entropy, values_pred = policy.evaluate_actions(mb_states, mb_actions, mb_masks, device=DEVICE)
                ratio = torch.exp(new_logps - old_logps_t[mb_idx])
                surr1 = ratio * advantages_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, returns_t[mb_idx])
                entropy_loss = -entropy.mean()
                loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 10 == 0:
            torch.save(policy.state_dict(), os.path.join(MODEL_DIR, f"policy_epoch{epoch+1}.pt"))
            # quick eval
            eval_env = env
            obs = eval_env.reset()
            done = False; tot = 0.0
            while not done:
                bx = flatten_obs(obs)
                logits_per_robot, _ = policy(bx)
                acts, _ = policy.sample_actions([l for l in logits_per_robot], [obs['masks']], device=DEVICE)
                action = acts[0].cpu().numpy()
                obs, r, done, info = eval_env.step(action)
                tot += r
            print(f"Epoch {epoch+1}/{PPO_EPOCHS}, eval return {tot:.2f}")

    torch.save(policy.state_dict(), os.path.join(MODEL_DIR, "policy_final.pt"))
    print("Training complete. Models saved in", MODEL_DIR)

if __name__ == "__main__":
    main()
