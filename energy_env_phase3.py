"""
energy_env_phase3.py
Phase 3: Cooperative multi-robot delivery environment with energy-aware load sharing.
Author: Rahul Kapardar
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiRobotEnergyEnvPhase3(gym.Env):
    """Environment supporting cooperative load sharing among robots."""

    metadata = {"render_modes": []}

    def __init__(self,
                 n_robots=3,
                 grid_size=5,
                 battery_init=50,
                 energy_threshold=15,
                 comm_range=2,
                 debug=True):
        super().__init__()

        # Parameters
        self.n_robots = n_robots
        self.grid_size = grid_size
        self.battery_init = battery_init
        self.energy_threshold = energy_threshold
        self.comm_range = comm_range
        self.debug = debug

        # Constants
        self.a = 0.1                # energy coefficient
        self.b = 0.1                # base consumption
        self.lambda_viol = 100      # violation penalty
        self.r_goal = 100           # delivery reward
        self.r_coop = 20            # cooperation reward

        # Gymnasium spaces
        self.action_space = spaces.MultiDiscrete([5] * n_robots)
        self.observation_space = spaces.Box(low=0, high=100,
                                            shape=(4 * n_robots,),
                                            dtype=np.float32)

        self.reset()

    # ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = []
        for i in range(self.n_robots):
            self.robots.append({
                "pos": np.array([i, 0]),
                "load": 1,
                "battery": float(self.battery_init),
                "destination": np.array([self.grid_size - 1, self.grid_size - 1])
            })
        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    # ----------------------------------------------------------
    def _get_obs(self):
        obs = []
        for r in self.robots:
            obs.extend([int(r["pos"][0]), int(r["pos"][1]), int(r["load"]), r["battery"]])
        return np.array(obs, dtype=np.float32)

    def _energy_cost(self, robot):
        return self.a * (robot["load"] ** 2) + self.b

    def _distance(self, r1, r2):
        return np.linalg.norm(r1["pos"] - r2["pos"], ord=1)

    def _find_nearest_helper(self, idx):
        """Find nearest free robot within communication range."""
        candidates = []
        for j, r in enumerate(self.robots):
            if j != idx and r["load"] == 0 and r["battery"] > self.energy_threshold:
                d = self._distance(self.robots[idx], r)
                if d <= self.comm_range:
                    candidates.append((j, d))
        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return None

    # ----------------------------------------------------------
    def step(self, actions):
        total_energy = 0.0
        coop_reward = 0.0
        violation_penalty = 0.0
        all_delivered = True

        # --- Cooperative phase: check low-battery robots
        for i, robot in enumerate(self.robots):
            if robot["battery"] < self.energy_threshold and robot["load"] == 1:
                helper = self._find_nearest_helper(i)
                if helper is not None:
                    self.robots[helper]["load"] = 1
                    robot["load"] = 0
                    coop_reward += self.r_coop
                    if self.debug:
                        print(f"🤝 Robot {i} transferred load to Robot {helper}")
                else:
                    violation_penalty += self.lambda_viol
                    if self.debug:
                        print(f"⚠️ Robot {i} requested help but no robot available")

        # --- Movement and energy updates
        for i, action in enumerate(actions):
            r = self.robots[i]
            # movement actions
            if action == 1 and r["pos"][1] < self.grid_size - 1:  # north
                r["pos"][1] += 1
            elif action == 2 and r["pos"][1] > 0:                 # south
                r["pos"][1] -= 1
            elif action == 3 and r["pos"][0] < self.grid_size - 1:  # east
                r["pos"][0] += 1
            elif action == 4 and r["pos"][0] > 0:                 # west
                r["pos"][0] -= 1

            # energy consumption
            cost = self._energy_cost(r)
            r["battery"] -= cost
            total_energy += cost

            # delivery check
            if np.array_equal(r["pos"], r["destination"]) and r["load"] == 1:
                r["load"] = 0
                coop_reward += self.r_goal
                if self.debug:
                    print(f"🎯 Robot {i} delivered parcel! +{self.r_goal}")

            if r["load"] == 1:
                all_delivered = False

        # --- Collision check
        positions = [tuple(r["pos"]) for r in self.robots]
        if len(positions) != len(set(positions)):
            violation_penalty += self.lambda_viol
            if self.debug:
                print("💥 Constraint violated: Collision")

        done = all_delivered or all(r["battery"] <= 0 for r in self.robots)
        reward = -total_energy - violation_penalty + coop_reward

        self.steps += 1
        if self.debug:
            print(f"Step {self.steps} | Action={actions} | Reward={reward:.2f} | Done={done}")

        obs = self._get_obs()
        return obs, reward, done, False, {}
