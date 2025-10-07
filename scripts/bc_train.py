"""
bc_train.py
-----------
Behavior Cloning (supervised imitation learning) for MultiRobotEnergyEnv.
Trains a neural policy that predicts joint robot actions from state inputs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class DemoDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.int64)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """Multi-head policy network: one action head per robot."""

    def __init__(self, state_dim, n_robots, hidden_sizes=[256, 256]):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(last, 5) for _ in range(n_robots)])

    def forward(self, x):
        h = self.shared(x)
        logits = [head(h) for head in self.heads]
        return torch.stack(logits, dim=1)  # (B, n_robots, 5)


def train_bc():
    data = np.load("data/demos.npz")
    states = data["states"]
    actions = data["actions"]
    n_robots = actions.shape[1]

    dataset = DemoDataset(states, actions)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCPolicy(states.shape[1], n_robots).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        model.train()
        total_loss = 0
        for s, a in loader:
            s, a = s.to(device), a.to(device)
            logits = model(s)
            loss = sum(criterion(logits[:, i, :], a[:, i]) for i in range(n_robots))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/15 - Loss: {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "state_dim": states.shape[1],
            "n_robots": n_robots,
        },
        "models/bc_policy.pt",
    )
    print("BC model saved to models/bc_policy.pt")


if __name__ == "__main__":
    train_bc()
