from energy_env_multi import MultiRobotEnergyEnv

env = MultiRobotEnergyEnv(n_robots=3, grid_size=5, battery=15,
                          a=1.0, b=0.1, r_goal=50, lambda_viol=10, debug=True)

obs, info = env.reset()
print("Initial state:", obs)

done = False
t = 0
while not done and t < 20:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {t+1} | action={action} | reward={reward:.2f} | done={done}")
    env.render()
    t += 1

print("Episode finished.")
