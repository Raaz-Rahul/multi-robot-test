from energy_env import SingleRobotEnergyEnv   # import your env

env = SingleRobotEnergyEnv(grid_size=5, battery=50)

obs, info = env.reset()
print("Initial State:", obs)

done = False
step_count = 0

while not done and step_count < 20:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Step {step_count+1}")
    print(f" Action taken: {action}")
    env.render()   
    print(f" Reward: {reward:.2f}, Done: {done}")
    print("-"*30)

    step_count += 1


print("Episode finished.")
