# test_phase3_env.py
"""
Phase 3 environment tests:
 - Observation structure
 - Energy reward equation 
 - Cooperative transfer logic
 - Delivery logic
 - Violations (battery, collision)
 - PPO compatibility test
"""

import numpy as np
from energy_env_phase3 import MultiRobotEnergyEnvPhase3
from stable_baselines3 import PPO


def test_reset():
    print("\n=== Test 1: Environment Reset ===")
    env = MultiRobotEnergyEnvPhase3(debug=True)
    obs, info = env.reset()
    print("Initial observation:", obs)

    assert obs.shape[0] == env.n_robots * 5, \
        "Observation dimension should be 5 features per robot"
    print("PASS ✔ Reset and observation shape correct")


def test_reward_equation():
    print("\n=== Test 2: Reward Formula ===")

    env = MultiRobotEnergyEnvPhase3(debug=True, shaped_reward=False)
    obs, _ = env.reset()

    actions = [0] * env.n_robots
    new_obs, reward, done, trunc, info = env.step(actions)

    # Expected energy cost:
    # rt = -sum_i (a*(W^2)*d + b)
    manual_cost = 0
    for i in range(env.n_robots):
        W = 1
        d = 0
        cost = env.a * (W ** 2) * d + env.b
        manual_cost += cost

    expected_reward = -manual_cost

    print(f"Env reward: {reward:.3f} | Expected: {expected_reward:.3f}")
    assert np.isclose(reward, expected_reward), "Reward formula mismatch"

    print("PASS ✔ Reward equation matches")


def test_transfer_logic():
    print("\n=== Test 3: Cooperative Transfer ===")

    env = MultiRobotEnergyEnvPhase3(enable_sharing=True, debug=True)
    obs, _ = env.reset()

    # Make robot 0 low battery, robot 1 high battery & idle
    env.robots[0]["battery"] = 5
    env.robots[1]["load"] = 0  # helper available

    actions = [5, 0, 0]  # robot 0 requests transfer
    new_obs, reward, done, trunc, info = env.step(actions)

    print("Transfer info:", info["transfers"])
    assert len(info["transfers"]) > 0, "No transfer happened"

    print("PASS ✔ Transfer logic works")


def test_delivery_logic():
    print("\n=== Test 4: Delivery ===")

    env = MultiRobotEnergyEnvPhase3(enable_sharing=False, debug=True)
    obs, _ = env.reset()

    # Teleport robot 0 to the destination with load = 1
    r0 = env.robots[0]
    dx, dy = r0["destination"]
    r0["x"], r0["y"] = dx, dy
    r0["load"] = 1

    actions = [0, 0, 0]
    _, reward, done, _, info = env.step(actions)

    print("Deliveries:", info["deliveries"])
    assert 0 in info["deliveries"], "Robot 0 should have delivered"

    print("PASS ✔ Delivery logic works")


def test_collision_violation():
    print("\n=== Test 5: Collision Violation ===")

    env = MultiRobotEnergyEnvPhase3(debug=True)
    obs, _ = env.reset()

    # Force collision: move robots to same coordinates
    env.robots[0]["x"] = 0
    env.robots[0]["y"] = 0
    env.robots[1]["x"] = 0
    env.robots[1]["y"] = 0

    actions = [0, 0, 0]
    _, reward, done, trunc, info = env.step(actions)

    print("Violation reasons:", info["violation_reasons"])
    assert "collision" in info["violation_reasons"], "Collision not detected!"

    print("PASS ✔ Collision violation detected")


def test_battery_violation():
    print("\n=== Test 6: Battery Violation ===")

    env = MultiRobotEnergyEnvPhase3(debug=True)
    obs, _ = env.reset()

    env.robots[0]["battery"] = -1  # force violation

    actions = [0, 0, 0]
    _, reward, done, trunc, info = env.step(actions)

    print("Violation reasons:", info["violation_reasons"])
    assert "battery_R0_neg" in info["violation_reasons"], "Battery violation not detected!"

    print("PASS ✔ Battery violation detected")


def test_phase2_energy_forecast_feature():
    print("\n=== Test 7: Energy Forecast Feature Ehat ===")

    env = MultiRobotEnergyEnvPhase3(debug=True)
    obs, _ = env.reset()

    obs = obs.reshape(env.n_robots, 5)
    Ehat_values = obs[:, 4]

    print("Ehat values:", Ehat_values)
    assert np.all(Ehat_values > 0), "Ehat must be positive"

    print("PASS ✔ Energy forecast feature exists and valid")


def test_ppo_compatibility():
    print("\n=== Test 8: PPO Compatibility ===")

    env = MultiRobotEnergyEnvPhase3()
    model = PPO("MlpPolicy", env, verbose=0)

    obs, _ = env.reset()
    action, _ = model.predict(obs)

    new_obs, reward, done, trunc, info = env.step(action.tolist())

    print("Step completed successfully.")
    print("Action:", action)

    print("PASS ✔ PPO can interact with Phase3 environment")


if __name__ == "__main__":
    test_reset()
    test_reward_equation()
    test_transfer_logic()
    test_delivery_logic()
    test_collision_violation()
    test_battery_violation()
    test_phase2_energy_forecast_feature()
    test_ppo_compatibility()

    print("\n====================")
    print("All Phase-3 tests passed successfully.")
    print("====================")
