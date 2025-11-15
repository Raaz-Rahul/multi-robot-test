import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from energy_env_phase3 import MultiRobotEnergyEnvPhase3


# ----------------------------- DEMO GENERATION ----------------------------- #

def generate_expert_demos(num_episodes=100, max_steps=200, save_path="data/phase3_demos.npz"):

    os.makedirs("data", exist_ok=True)
    env = MultiRobotEnergyEnvPhase3(debug=False)
    demos = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        for t in range(max_steps):

            # simple expert heuristic: move toward destination
            actions = []
            for r in range(env.n_robots):
                x, y, load, battery = obs[r*4:(r*4)+4]
                dx, dy = env.robots[r]["destination"]

                if x < dx: act = 3
                elif x > dx: act = 4
                elif y < dy: act = 1
                else: act = 0

                actions.append(act)

            next_obs, reward, done, _, info = env.step(actions)
            demos.append((obs, actions, reward))
            obs = next_obs
            if done:
                break

    np.savez(save_path, demos=demos)
    print(f"✔ Expert demos saved to {save_path}")
    return save_path


# ----------------------------- TRAINING PPO ----------------------------- #

def train_ppo_phase3():

    print("🚀 Phase-3 PPO Training Started...")

    env = MultiRobotEnergyEnvPhase3(debug=False)
    env = Monitor(env)                     # record episode reward/length
    env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                verbose=1)

    os.makedirs("models", exist_ok=True)

    model.learn(total_timesteps=150_000)

    model.save("models/ppo_phase3.zip")
    print("✔ PPO model saved: models/ppo_phase3.zip")

    return model


# ----------------------------- MAIN PIPELINE ----------------------------- #

if __name__ == "__main__":

    # 1) Generate demos (optional but useful)
    generate_expert_demos()

    # 2) Train PPO
    train_ppo_phase3()

    print("\n🎉 Phase-3 Training Completed Successfully!\n")
