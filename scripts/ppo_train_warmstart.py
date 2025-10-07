"""
ppo_train_warmstart.py
----------------------
Warm-starts a PPO agent with Behavior-Cloning weights and trains it further
on MultiRobotEnergyEnv.
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_robot import MultiRobotEnergyEnv


def make_env():
    def _init():
        return MultiRobotEnergyEnv(n_robots=3, grid_size=5, battery=30, debug=False)
    return _init


def warmstart_policy(model, bc_path):
    if not os.path.exists(bc_path):
        print("⚠️ BC checkpoint not found, skipping warm-start.")
        return
    bc = torch.load(bc_path, map_location=model.device)
    bc_state = bc["state_dict"]
    policy_state = model.policy.state_dict()

    # copy weights if shape matches
    for k_bc, v_bc in bc_state.items():
        for k_pol in list(policy_state.keys()):
            if v_bc.shape == policy_state[k_pol].shape:
                policy_state[k_pol] = v_bc
                break
    model.policy.load_state_dict(policy_state)
    print("✅ Warm-started PPO with BC weights.")


def main():
    env = DummyVecEnv([make_env() for _ in range(4)])
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=8192,
        n_epochs=4,
        policy_kwargs=policy_kwargs,
    )

    warmstart_policy(model, "models/bc_policy.pt")

    model.learn(total_timesteps=80000)
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_multi_robot")
    print("✅ PPO model saved to models/ppo_multi_robot.zip")


if __name__ == "__main__":
    main()
