import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env_multi_robot import MultiRobotEnv
from policy import CentralizedPolicy


# ---------- Heuristic per Algorithm 1 (phase 3) ---------- #
# Uses:
#  - offload candidates R-:  ̂E_i > beta * B_i
#  - onload candidates  R+:  ̂E_i <= beta * B_i
#  - partner set Cand(i): j in R+ with Δ(p_i, p_j) <= 2r
#  - meeting node M: midpoint on grid within radius r of both
#  - share amount ω: floor((W_i - W_j)/2) clipped to feasibility
# [attached_file:1]


def heuristic_labels(env):
    """
    Given current env state, produce a joint action dict:
    - Movement: greedy towards destination.
    - Sharing heads from Algorithm 1 heuristic. [attached_file:1]
    """
    n = env.num_robots
    move = np.zeros(n, dtype=np.int64)
    offload = np.zeros(n, dtype=np.int64)
    partner = np.zeros(n, dtype=np.int64)
    meeting = np.zeros(n, dtype=np.int64)
    amount = np.zeros(n, dtype=np.int64)

    # 1) Greedy movement: each robot moves along its planned shortest path. [attached_file:1]
    for i in range(n):
        if len(env.routes[i]) == 0:
            move[i] = 4  # Stay
        else:
            nx, ny = env.routes[i][0]
            x, y = env.pos[i]
            if nx < x:
                move[i] = 0  # N
            elif nx > x:
                move[i] = 1  # S
            elif ny > y:
                move[i] = 2  # E
            elif ny < y:
                move[i] = 3  # W
            else:
                move[i] = 4

    # 2) Offload and onload candidate sets using forecast and beta threshold. [attached_file:1]
    R_minus = []  # offload
    R_plus = []   # onload
    for i in range(n):
        if env.forecast[i] > env.beta * env.battery_budget:
            R_minus.append(i)
        else:
            R_plus.append(i)

    # 3) For each i in R-, choose partner j in R+ within 2r minimizing projected team energy. [attached_file:1]
    def manhattan(u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])

    used_receivers = set()
    for i in R_minus:
        cand = [j for j in R_plus if j not in used_receivers and manhattan(env.pos[i], env.pos[j]) <= 2 * env.share_radius]
        if len(cand) == 0:
            continue

        # Simple choice: pick nearest j (proxy for lower energy) [attached_file:1]
        j = min(cand, key=lambda jj: manhattan(env.pos[i], env.pos[jj]))

        # 4) Choose meeting node M within radius r of both; use approximate midpoint clipped to grid. [attached_file:1]
        pi = env.pos[i]
        pj = env.pos[j]
        mx = int(round((pi[0] + pj[0]) / 2))
        my = int(round((pi[1] + pj[1]) / 2))
        mx = max(0, min(env.H - 1, mx))
        my = max(0, min(env.W - 1, my))

        if manhattan(pi, (mx, my)) > env.share_radius or manhattan(pj, (mx, my)) > env.share_radius:
            # fallback: just use receiver's position as meeting node if within r. [attached_file:1]
            mx, my = pj
            if manhattan(pi, (mx, my)) > env.share_radius:
                continue

        # 5) Share amount ω = floor((W_i - W_j)/2) clipped to capacity. [attached_file:1]
        wi = int(env.load[i])
        wj = int(env.load[j])
        w = max(0, (wi - wj) // 2)
        if w <= 0:
            continue
        cap_j = env.max_load - wj
        w = min(w, wi, cap_j)
        if w <= 0:
            continue

        offload[i] = 1
        partner[i] = j
        meeting[i] = mx * env.W + my
        amount[i] = w
        used_receivers.add(j)

    return {
        "move": move,
        "offload": offload,
        "partner": partner,
        "meeting": meeting,
        "amount": amount,
    }


# ---------- Behavior cloning dataset generator ---------- #


def generate_dataset(env, policy_obs_dim, num_episodes=200, max_steps=200):
    """
    Roll out heuristic policy to build (s, a) dataset for behavior cloning. [attached_file:1]
    """
    obs_list = []
    act_list = []

    for ep in range(num_episodes):
        env.phase = 3  # ensure forecast + sharing heads enabled in observation. [attached_file:1]
        obs, info = env.reset()
        for t in range(max_steps):
            labels = heuristic_labels(env)
            obs_list.append(obs.copy())
            act_list.append(labels)
            obs, reward, done, truncated, info = env.step(labels)
            if done or truncated:
                break

    obs_arr = np.stack(obs_list, axis=0)
    return obs_arr, act_list


# ---------- Behavior cloning training ---------- #


def to_tensor_action(actions, device):
    keys = ["move", "offload", "partner", "meeting", "amount"]
    out = {}
    B = len(actions)
    for k in keys:
        arr = np.stack([a[k] for a in actions], axis=0)
        out[k] = torch.from_numpy(arr).long().to(device)
    return out


def bc_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 3 environment: same as before but phase set to 3. [attached_file:1]
    env = MultiRobotEnv(
        grid_size=(10, 10),
        num_robots=3,
        phase=3,
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

    # 1) Generate heuristic dataset D using Algorithm 1. [attached_file:1]
    print("Generating heuristic dataset...")
    obs_arr, act_list = generate_dataset(env, obs_dim, num_episodes=300, max_steps=200)
    actions_tensor = to_tensor_action(act_list, device)
    obs_tensor = torch.from_numpy(obs_arr).float().to(device)

    dataset_size = obs_tensor.size(0)
    batch_size = 1024
    epochs = 10

    # 2) Behavior Cloning loss: L_BC = -E_{(s,a)~D} log πθ(a|s) (Eq. 5). [attached_file:1]
    for epoch in range(epochs):
        perm = np.random.permutation(dataset_size)
        epoch_loss = 0.0
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = perm[start:end]
            batch_obs = obs_tensor[idx]
            batch_action = {k: v[idx] for k, v in actions_tensor.items()}

            masks = None  # heuristic labels already feasible; can optionally include masks. [attached_file:1]
            logp, entropy, value = policy.evaluate_actions(batch_obs, batch_action, masks)

            loss = -logp.mean()  # behavior cloning objective

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        print(f"[BC] epoch {epoch+1}/{epochs} loss={epoch_loss / dataset_size:.4f}")

    # 3) Save BC-pretrained weights for PPO warm start. [attached_file:1]
    torch.save(policy.state_dict(), "centralized_policy_bc.pt")
    print("Saved BC policy to centralized_policy_bc.pt")


if __name__ == "__main__":
    bc_train()
