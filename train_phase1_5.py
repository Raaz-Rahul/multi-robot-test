import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env_multi_robot import MultiRobotEnv
from policy1_5 import CentralizedPolicy



def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + values[:T]
    return adv, returns


def to_tensor_action(batch_actions, device):
    # Convert list of dicts into dict of tensors [B, n]
    keys = ["move", "offload", "partner", "meeting", "amount"]
    out = {}
    B = len(batch_actions)
    for k in keys:
        arr = np.stack([a[k] for a in batch_actions], axis=0)
        out[k] = torch.from_numpy(arr).long().to(device)
    return out


def make_mask_batch(mask_list, env, device):
    # Each entry in mask_list is env._compute_masks() output.
    B = len(mask_list)
    n = env.num_robots
    movement = np.stack([m["movement"] for m in mask_list], axis=0)
    partner = np.stack([m["partner"] for m in mask_list], axis=0)
    meeting = np.stack([m["meeting"] for m in mask_list], axis=0)
    amount = np.stack([m["amount"] for m in mask_list], axis=0)
    return {
        "movement": torch.from_numpy(movement).float().to(device),
        "partner": torch.from_numpy(partner).float().to(device),
        "meeting": torch.from_numpy(meeting).float().to(device),
        "amount": torch.from_numpy(amount).float().to(device),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiRobotEnv(
        grid_size=(10, 10),
        num_robots=3,
        phase=1,
        max_steps=200,
    )

    obs_dim = env.obs_dim
    policy = CentralizedPolicy(
        obs_dim=obs_dim,
        num_robots=env.num_robots,
        num_move=env.num_move,
        num_offload=env.num_offload,
        num_partner=env.num_partner,
        num_meeting=env.num_meeting,
        num_amount=env.num_amount,
        hidden_dim=256,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    # Hyperparameters from Table 1 (approx). [attached_file:1]
    total_steps = 200_000
    rollout_len = 2048
    ppo_epochs = 4
    mini_batch_size = 1024
    clip_eps = 0.1
    ent_coef = 0.01
    kl_coef = 0.5
    vf_coef = 0.5
    gamma = 0.99
    lam = 0.95

    # Curriculum (Section 6): steps at which phase increments. [attached_file:1]
    boundaries = {
        1: 0,
        2: 40_000,
        3: 70_000,
        4: 100_000,
        5: 130_000,
    }

    def update_phase(global_step):
        phase = 1
        for p, s in sorted(boundaries.items(), key=lambda x: x[1]):
            if global_step >= s:
                phase = p
        env.phase = phase
        return phase

    global_step = 0
    obs, info = env.reset()
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    while global_step < total_steps:
        current_phase = update_phase(global_step)

        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []
        mask_buf = []

        for _ in range(rollout_len):
            masks_np = info["masks"]
            masks_t = {
                "movement": torch.from_numpy(masks_np["movement"]).float().unsqueeze(0).to(device),
                "partner": torch.from_numpy(masks_np["partner"]).float().unsqueeze(0).to(device),
                "meeting": torch.from_numpy(masks_np["meeting"]).float().unsqueeze(0).to(device),
                "amount": torch.from_numpy(masks_np["amount"]).float().unsqueeze(0).to(device),
            }

            with torch.no_grad():
                action, logp, entropy, value = policy.sample_action(obs_t, masks=masks_t)

            next_obs, reward, done, truncated, info = env.step(action)
            global_step += 1

            obs_buf.append(obs_t.cpu().numpy())
            act_buf.append(action)
            rew_buf.append(reward)
            done_buf.append(done or truncated)
            val_buf.append(value.item())
            logp_buf.append(logp.item())
            mask_buf.append(masks_np)

            obs = next_obs
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            if done or truncated:
                obs, info = env.reset()
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

            if global_step >= total_steps:
                break

        # GAE
        rewards = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.bool_)
        values = np.array(val_buf + [0.0], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, gamma, lam)

        obs_batch = torch.from_numpy(np.concatenate(obs_buf, axis=0)).float().to(device)
        act_batch = to_tensor_action(act_buf, device)
        adv_batch = torch.from_numpy(adv).float().to(device)
        ret_batch = torch.from_numpy(ret).float().to(device)
        logp_old_batch = torch.from_numpy(np.array(logp_buf, dtype=np.float32)).float().to(device)
        mask_batch = make_mask_batch(mask_buf, env, device)

        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        n_samples = obs_batch.size(0)
        indices = np.arange(n_samples)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                mb_obs = obs_batch[mb_idx]
                mb_adv = adv_batch[mb_idx]
                mb_ret = ret_batch[mb_idx]
                mb_logp_old = logp_old_batch[mb_idx]

                mb_action = {k: v[mb_idx] for k, v in act_batch.items()}
                mb_masks = {
                    "movement": mask_batch["movement"][mb_idx],
                    "partner": mask_batch["partner"][mb_idx],
                    "meeting": mask_batch["meeting"][mb_idx],
                    "amount": mask_batch["amount"][mb_idx],
                }

                logp, entropy, value = policy.evaluate_actions(mb_obs, mb_action, masks=mb_masks)

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, mb_ret)
                entropy_loss = -entropy.mean()
                kl_est = torch.mean((mb_logp_old - logp) ** 2)  # simple surrogate

                loss = (
                    policy_loss
                    + vf_coef * value_loss
                    + ent_coef * entropy_loss
                    + kl_coef * kl_est
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        print(
            f"step={global_step} phase={current_phase} "
            f"avgR={rewards.mean():.2f} "
            f"advMean={adv.mean():.2f}"
        )


if __name__ == "__main__":
    main()
