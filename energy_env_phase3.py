# energy_env_phase3.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnvPhase3(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_robots=3, grid_size=5, battery_init=50.0, debug=False):
        super().__init__()
        self.n_robots = int(n_robots)
        self.grid_size = int(grid_size)
        self.battery_init = float(battery_init)
        self.debug = bool(debug)

        # 6 actions per robot: 0 stay,1 north,2 south,3 east,4 west,5 request transfer
        self.action_space = spaces.MultiDiscrete([6] * self.n_robots)

        # Observation: [x,y,load,battery,Ehat] per robot
        high = np.array([self.grid_size, self.grid_size, 1.0, self.battery_init, self.battery_init] * self.n_robots, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, dtype=np.float32)

        # energy & reward params (as approved)
        self.a = 0.1
        self.b = 0.1
        self.lambda_viol = 10.0
        self.r_goal = 100.0
        self.coop_bonus = 20.0

        # sharing / thresholds
        self.low_battery_threshold = 10.0
        self.comm_range = 2

        self.robots = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = []
        for i in range(self.n_robots):
            self.robots.append({
                "x": int(i),
                "y": 0,
                "load": 1,
                "battery": float(self.battery_init),
                "destination": (self.grid_size - 1, self.grid_size - 1)
            })
        return self._get_obs(), {}

    def _estimate_future_energy(self, r):
        dx, dy = r["destination"]
        dist = abs(dx - r["x"]) + abs(dy - r["y"])
        per_step = self.a * (r["load"] ** 2) * 1 + self.b
        return dist * per_step

    def _get_obs(self):
        flat = []
        for r in self.robots:
            Ehat = self._estimate_future_energy(r)
            flat.extend([float(r["x"]), float(r["y"]), float(r["load"]), float(r["battery"]), float(Ehat)])
        return np.array(flat, dtype=np.float32)

    def step(self, actions):
        assert len(actions) == self.n_robots, f"Expected {self.n_robots} actions"

        # store old positions to compute movement indicator d_i
        old_pos = [(r["x"], r["y"]) for r in self.robots]

        # apply movement for actions 0..4; 5=no movement (request transfer)
        for i, a in enumerate(actions):
            r = self.robots[i]
            x, y = r["x"], r["y"]
            if a == 1 and y < self.grid_size - 1:
                y += 1
            elif a == 2 and y > 0:
                y -= 1
            elif a == 3 and x < self.grid_size - 1:
                x += 1
            elif a == 4 and x > 0:
                x -= 1
            # else stay or request transfer
            r["x"], r["y"] = int(x), int(y)

        # movement indicator per robot (1 if moved)
        d_list = [1 if (r["x"], r["y"]) != old_pos[i] else 0 for i, r in enumerate(self.robots)]

        # energy cost & battery update
        energy_costs = []
        for i, r in enumerate(self.robots):
            W = r["load"]
            di = d_list[i]
            cost = self.a * (W ** 2) * di + self.b
            r["battery"] -= cost
            energy_costs.append(cost)

        # handle transfer requests (action == 5)
        transfers = []
        coop_reward = 0.0
        for i, a in enumerate(actions):
            if a == 5:
                req = self.robots[i]
                if req["load"] == 1:
                    helper_idx = None
                    for j, h in enumerate(self.robots):
                        if j == i:
                            continue
                        if h["load"] == 0 and h["battery"] > self.low_battery_threshold:
                            dist = abs(req["x"] - h["x"]) + abs(req["y"] - h["y"])
                            if dist <= self.comm_range:
                                helper_idx = j
                                break
                    if helper_idx is not None:
                        req["load"] = 0
                        self.robots[helper_idx]["load"] = 1
                        transfers.append((i, helper_idx))
                        coop_reward += self.coop_bonus
                        if self.debug:
                            print(f"R{i} requested transfer → R{helper_idx} (+{self.coop_bonus})")

        # automatic transfer if low battery and helper idle nearby
        for i in range(self.n_robots):
            ri = self.robots[i]
            if ri["load"] == 1 and ri["battery"] < self.low_battery_threshold:
                for j in range(self.n_robots):
                    if i == j: continue
                    rj = self.robots[j]
                    if rj["load"] == 0 and rj["battery"] > self.low_battery_threshold:
                        d = abs(ri["x"] - rj["x"]) + abs(ri["y"] - rj["y"])
                        if d <= self.comm_range:
                            ri["load"] = 0
                            rj["load"] = 1
                            transfers.append((i, j))
                            coop_reward += self.coop_bonus
                            if self.debug:
                                print(f"R{i} auto-transfer → R{j} (+{self.coop_bonus})")
                            break

        # deliveries
        deliveries = []
        for i, r in enumerate(self.robots):
            if r["load"] == 1:
                dx, dy = r["destination"]
                if (r["x"], r["y"]) == (dx, dy):
                    r["load"] = 0
                    deliveries.append(i)
                    if self.debug:
                        print(f"R{i} delivered parcel! +{self.r_goal}")

        # violations: battery < 0 and collisions
        violation = False
        violation_reasons = []
        for i, r in enumerate(self.robots):
            if r["battery"] < 0.0:
                violation = True
                violation_reasons.append(f"battery_R{i}_neg")
                if self.debug:
                    print(f"Constraint violated: R{i} battery < 0")

        positions = [(r["x"], r["y"]) for r in self.robots]
        if len(positions) != len(set(positions)):
            violation = True
            violation_reasons.append("collision")
            if self.debug:
                print("Constraint violated: Collision")

        all_delivered = all(r["load"] == 0 for r in self.robots)
        all_dead = all(r["battery"] <= 0.0 for r in self.robots)
        done = all_delivered or all_dead

        # reward components
        r_energy = -float(sum(energy_costs))
        r_viol = -self.lambda_viol if violation else 0.0
        r_goal = float(self.r_goal) if all_delivered else 0.0
        r_coop = float(coop_reward)

        reward = r_energy + r_viol + r_goal + r_coop

        info = {
            "energy_costs": energy_costs,
            "d_list": d_list,
            "transfers": transfers,
            "deliveries": deliveries,
            "violation": violation,
            "violation_reasons": violation_reasons,
            "r_energy": r_energy,
            "r_viol": r_viol,
            "r_goal": r_goal,
            "r_coop": r_coop
        }

        return self._get_obs(), float(reward), bool(done), False, info
