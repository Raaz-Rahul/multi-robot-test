# policy.py
# Mask-aware actor-critic for MultiRobotEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

def torch_mask_from_np(mask_np, device=None):
    return torch.tensor(mask_np, dtype=torch.bool, device=device)

class MaskedPolicy(nn.Module):
    """
    Shared encoder with per-robot factorized heads. Utility methods:
      - forward(obs) -> (logits_per_robot, values)
      - sample_actions(logits_per_robot, masks_batch)
      - evaluate_actions(obs, actions, masks_batch)
    """
    def __init__(self, obs_dim: int, n_robots: int, per_robot_dims: List[int], hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.n = n_robots
        self.per_robot_dims = per_robot_dims
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(hidden, sum(per_robot_dims)) for _ in range(n_robots)])
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        logits_per_robot = [head(h) for head in self.heads]
        values = self.critic(h).squeeze(-1)
        return logits_per_robot, values

    def sample_actions(self, logits_per_robot: List[torch.Tensor], masks_batch: List[Dict], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = logits_per_robot[0].shape[0]
        actions = torch.zeros(batch, self.n * len(self.per_robot_dims), dtype=torch.int64, device=device)
        logps = torch.zeros(batch, dtype=torch.float32, device=device)
        for b in range(batch):
            out = []
            lp = 0.0
            for i in range(self.n):
                logits = logits_per_robot[i][b]
                start = 0
                for hid, size in enumerate(self.per_robot_dims):
                    seg = logits[start:start+size].clone()
                    key = ['move_mask','offload_mask','partner_mask','meeting_mask','w_mask'][hid]
                    mask_np = masks_batch[b][i][key]
                    mask_t = torch_mask_from_np(mask_np, device=device)
                    if seg.shape[0] != mask_t.shape[0]:
                        if mask_t.shape[0] < seg.shape[0]:
                            pad = seg.shape[0] - mask_t.shape[0]
                            mask_t = torch.cat([mask_t, torch.ones(pad, dtype=torch.bool, device=device)])
                        else:
                            mask_t = mask_t[:seg.shape[0]]
                    seg[~mask_t] = -1e9
                    probs = F.softmax(seg, dim=0)
                    cat = torch.distributions.Categorical(probs)
                    a = cat.sample()
                    lp = lp + cat.log_prob(a)
                    out.append(int(a.item()))
                    start += size
            actions[b] = torch.tensor(out, dtype=torch.int64, device=device)
            logps[b] = lp
        return actions, logps

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, masks_batch: List[Dict], device=None):
        logits_per_robot, values = self.forward(obs)
        batch = obs.shape[0]
        logps = torch.zeros(batch, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch, dtype=torch.float32, device=device)
        for b in range(batch):
            lp = 0.0; ent = 0.0
            for i in range(self.n):
                logits = logits_per_robot[i][b]
                start = 0
                for hid, size in enumerate(self.per_robot_dims):
                    seg = logits[start:start+size].clone()
                    key = ['move_mask','offload_mask','partner_mask','meeting_mask','w_mask'][hid]
                    mask_np = masks_batch[b][i][key]
                    mask_t = torch_mask_from_np(mask_np, device=device)
                    if seg.shape[0] != mask_t.shape[0]:
                        if mask_t.shape[0] < seg.shape[0]:
                            pad = seg.shape[0] - mask_t.shape[0]
                            mask_t = torch.cat([mask_t, torch.ones(pad, dtype=torch.bool, device=device)])
                        else:
                            mask_t = mask_t[:seg.shape[0]]
                    seg[~mask_t] = -1e9
                    probs = F.softmax(seg, dim=0)
                    cat = torch.distributions.Categorical(probs)
                    act = int(actions[b, i*len(self.per_robot_dims) + hid].item())
                    if act >= seg.shape[0]:
                        lp += -1e3
                    else:
                        lp += cat.log_prob(torch.tensor(act, device=device))
                        ent += cat.entropy()
                    start += size
            logps[b] = lp
            entropy[b] = ent
        return logps, entropy, values
