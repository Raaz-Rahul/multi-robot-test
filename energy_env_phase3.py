import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnvPhase3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_robots=3, grid_size=5, battery_init=50, debug=False):
        super().__init__()

        self.n_robots = n_robots
        self.grid_size = grid_size
        self.debug = debug

        self.action_space = spaces.MultiDiscrete([5] * n_robots)

        high = np.array([grid_size, grid_size, 1, battery_init] * n_robots, dtype=float)
        self.observation_space = spaces.Box(0, high, dtype=float)

        self.battery_init = battery_init
        self.a = 0.1
        self.b = 0.1
        self.lambda_violation = 10
        self.coop_bonus = 20

        self.low_battery_threshold = 10
        self.comm_range = 2

        self.robots = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robots = []
        for i in range(self.n_robots):
            self.robots.append({
                "x": i,
                "y": 0,
                "load": 1,
                "battery": float(self.battery_init),
                "destination": (self.grid_size - 1, self.grid_size - 1)
            })

        return self._get_obs(), {}

    # -----------------------------------------------------

    def step(self, actions):
        assert len(actions) == self.n_robots

        # MOVE -----------------------------------------------------
        for i, act in enumerate(actions):
            robot = self.robots[i]
            x, y = robot["x"], robot["y"]

            if act == 1 and y < self.grid_size - 1:      # north
                y += 1
            elif act == 2 and y > 0:                    # south
                y -= 1
            elif act == 3 and x < self.grid_size - 1:   # east
                x += 1
            elif act == 4 and x > 0:                    # west
                x -= 1

            robot["x"], robot["y"] = x, y

        # ENERGY COST ----------------------------------------------
        energy_costs = []
        for i in range(self.n_robots):
            load = self.robots[i]["load"]
            d = 1  # distance per step = 1
            cost = self.a * (load ** 2) * d + self.b
            self.robots[i]["battery"] -= cost
            energy_costs.append(cost)

        # COOPERATIVE LOAD SHARING ----------------------------------
        coop_reward = 0
        for i in range(self.n_robots):
            r_i = self.robots[i]
            if r_i["load"] == 1 and r_i["battery"] < self.low_battery_threshold:
                for j in range(self.n_robots):
                    if i == j: continue
                    r_j = self.robots[j]
                    if r_j["load"] == 0:
                        if abs(r_i["x"] - r_j["x"]) + abs(r_i["y"] - r_j["y"]) <= self.comm_range:
                            r_i["load"] = 0
                            r_j["load"] = 1
                            coop_reward += self.coop_bonus
                            if self.debug:
                                print(f"Load transferred: R{i} → R{j} (+{self.coop_bonus})")
                            break

        # CHECK DELIVERY ---------------------------------------------
        delivery_reward = 0
        done = False

        for i in range(self.n_robots):
            r = self.robots[i]
            if r["load"] == 1:
                dx, dy = r["destination"]
                if r["x"] == dx and r["y"] == dy:
                    delivery_reward += 100
                    r["load"] = 0
                    if self.debug:
                        print(f"Robot {i} delivered parcel! +100")

        # CHECK VIOLATIONS -----------------------------------------
        violation = False
        violation_penalty = 0

        for i in range(self.n_robots):
            if self.robots[i]["battery"] < 0:
                violation = True
                violation_penalty -= self.lambda_violation
                if self.debug:
                    print(f"Battery violation: Robot {i}")

        positions = [(r["x"], r["y"]) for r in self.robots]
        if len(positions) != len(set(positions)):
            violation = True
            violation_penalty -= self.lambda_violation
            if self.debug:
                print("Collision detected!")

        # TERMINATION RULES -----------------------------------------
        all_delivered = all(r["load"] == 0 for r in self.robots)
        all_dead = all(r["battery"] <= 0 for r in self.robots)
        done = all_delivered or all_dead

        # TOTAL REWARD ----------------------------------------------
        reward = (
            -sum(energy_costs)
            + delivery_reward
            + coop_reward
            + violation_penalty
        )

        return self._get_obs(), reward, done, False, {}

    # -----------------------------------------------------

    def _get_obs(self):
        flat = []
        for r in self.robots:
            flat.extend([float(r["x"]), float(r["y"]), float(r["load"]), float(r["battery"])])
        return np.array(flat, dtype=float)
