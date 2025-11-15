import numpy as np
from stable_baselines3 import PPO
from energy_env_phase3 import MultiRobotEnergyEnvPhase3


def run_phase3_evaluation(episodes=3, max_steps=400):

    model_path = "models/ppo_phase3.zip"
    print(f"Loading PPO model: {model_path}")
    model = PPO.load(model_path)

    env = MultiRobotEnergyEnvPhase3(debug=True)

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        print(f"\n=================== EPISODE {ep+1} =====================")

        for t in range(max_steps):

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action.tolist())

            total_reward += reward

            print(f"Step {t} | Action={action.tolist()} | Reward={reward:.2f} | Done={done}")
            print(f"   Step info: transfers={info['transfers']}, deliveries={info['deliveries']}")

            if done:
                break

        print(f"=== Episode {ep+1} finished in {t+1} steps | reward {total_reward:.2f} ===")

    print("\n==================== Evaluation Complete ====================\n")


if __name__ == "__main__":
    run_phase3_evaluation()
