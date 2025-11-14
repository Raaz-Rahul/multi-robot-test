# energy_env_phase3.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnvPhase3(gym.Env):
    """
    Multi-robot energy environment (Phases 0-3).
    - Reward follows the formula in the attached PDF:
      r_t = - sum_i ( a*(W_i^t)^2 * d_i^t + b ) - lambda_viol * 1{violation} + r_goal * 1{all delivered}
    - Phase2: Adds energy-forecast Ehat_i^t to observations.
    - enable_sharing toggles cooperative transfer logic (use False for Phase1).
    - Optionally enable dense shaping (for faster learning) via shaped_reward=True.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_robots: int = 3,
        grid_size: int = 5,
        battery_init: float = 50.0,
        enable_sharing: bool = True,
        shaped_reward: bool = False,
        a: float = 0.1,
        b: float = 0.1,
        lambda_viol: float = 10.0,
        r_goal: float = 100.0,
        low_battery_threshold: float = 10.0,
        comm_range: int = 2,
        debug: bool = False,
    ):
        super().__init__()

        self.n_robots = int(n_robots)
        self.grid_size = int(grid_size)
        self.battery_init = float(battery_init)
        self.enable_sharing = bool(enable_sharing)
        self.shaped_reward = bool(shaped_reward)
        self.debug = bool(debug)

        # energy parameters (match PDF)
        self.a = float(a)
        self.b = float(b)
        self.lambda_viol = float(lambda_viol)
        self.r_goal = float(r_goal)

        # cooperation params
        self.low_battery_threshold = float(low_battery_threshold)
        self.comm_range = int(comm_range)

        # action space: 0 stay, 1 north, 2 south, 3 east, 4 west, 5 request transfer
        self.action_space = spaces.MultiDiscrete([6] * self.n_robots)

        # observation per robot: x, y, load (0/1), battery (float), Ehat (float)
        high = np.array(
            [self.grid_size, self.grid_size, 1, self.battery_init, self.battery_init] * self.n_robots,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(0.0, high, dtype=np.float32)

        self.robots = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = []
        # Place robots at x=i, y=0, load=1, battery=battery_init
        for i in range(self.n_robots):
            self.robots.append(
                {
                    "x": int(i),
                    "y": 0,
                    "load": 1,
                    "battery": float(self.battery_init),
                    "destination": (self.grid_size - 1, self.grid_size - 1),
                }
            )
        return self._get_obs(), {}

    def _estimate_future_energy(self, robot):
        """
        Phase-2 feature: Ehat_i^t = estimated energy needed to reach destination from current pos
        We use Manhattan distance * expected per-step energy assuming current load.
        """
        x, y, load, _ = robot["x"], robot["y"], robot["load"], robot["battery"]
        dest_x, dest_y = robot["destination"]
        dist = abs(dest_x - x) + abs(dest_y - y)
        per_step_cost = self.a * ((load) ** 2) * 1 + self.b
        return dist * per_step_cost

    def _get_obs(self):
        flat = []
        for r in self.robots:
            Ehat = self._estimate_future_energy(r)
            flat.extend([float(r["x"]), float(r["y"]), float(r["load"]), float(r["battery"]), float(Ehat)])
        return np.array(flat, dtype=np.float32)

    def step(self, actions):
        assert len(actions) == self.n_robots
        # record previous states to compute per-robot d_i^t (distance moved)
        old_positions = [(r["x"], r["y"]) for r in self.robots]
        old_states = [(r["x"], r["y"], r["load"], r["battery"]) for r in self.robots]

        # apply movements (0..4). action 5 is request-transfer and doesn't move
        for i, act in enumerate(actions):
            r = self.robots[i]
            x, y = r["x"], r["y"]
            if act == 1 and y < self.grid_size - 1:
                y += 1
            elif act == 2 and y > 0:
                y -= 1
            elif act == 3 and x < self.grid_size - 1:
                x += 1
            elif act == 4 and x > 0:
                x -= 1
            # else stay or transfer request
            r["x"], r["y"] = int(x), int(y)

        # compute per-robot distance moved (d_i^t). here discrete: 1 if moved else 0
        d_list = []
        for i, r in enumerate(self.robots):
            oldx, oldy = old_positions[i]
            moved = 1 if (r["x"] != oldx or r["y"] != oldy) else 0
            d_list.append(int(moved))

        # energy consumption per robot: a*(W^2)*d + b  (W is load, d is movement flag)
        energy_costs = []
        for i, r in enumerate(self.robots):
            W = r["load"]
            di = d_list[i]
            cost = self.a * (W ** 2) * di + self.b
            energy_costs.append(cost)
            r["battery"] -= cost

        # handle sharing if enabled
        transfers = []
        if self.enable_sharing:
            # explicit transfer requests (action==5) first
            for i, act in enumerate(actions):
                if act == 5:
                    req = self.robots[i]
                    if req["load"] == 1:
                        # find nearest idle helper with battery > threshold
                        nearest = None
                        nearest_dist = 1e9
                        for j, helper in enumerate(self.robots):
                            if j == i:
                                continue
                            if helper["load"] == 0 and helper["battery"] > self.low_battery_threshold:
                                d = abs(req["x"] - helper["x"]) + abs(req["y"] - helper["y"])
                                if d <= self.comm_range and d < nearest_dist:
                                    nearest = j
                                    nearest_dist = d
                        if nearest is not None:
                            self.robots[i]["load"] = 0
                            self.robots[nearest]["load"] = 1
                            transfers.append((i, nearest))
            # also automatic transfer: low battery robots can offload to nearby idle robots
            for i in range(self.n_robots):
                ri = self.robots[i]
                if ri["load"] == 1 and ri["battery"] < self.low_battery_threshold:
                    for j in range(self.n_robots):
                        if i == j:
                            continue
                        rj = self.robots[j]
                        if rj["load"] == 0 and rj["battery"] > self.low_battery_threshold:
                            d = abs(ri["x"] - rj["x"]) + abs(ri["y"] - rj["y"])
                            if d <= self.comm_range:
                                ri["load"] = 0
                                rj["load"] = 1
                                transfers.append((i, j))
                                break

        # deliveries: when a robot with load==1 is at its destination -> set load to 0
        deliveries = []
        for i, r in enumerate(self.robots):
            if r["load"] == 1:
                dx, dy = r["destination"]
                if (r["x"], r["y"]) == (dx, dy):
                    r["load"] = 0
                    deliveries.append(i)

        # violations: battery<0 or collision
        violation = False
        violation_reasons = []
        for i, r in enumerate(self.robots):
            if r["battery"] < 0.0:
                violation = True
                violation_reasons.append(f"battery_R{i}_neg")
        positions = [(r["x"], r["y"]) for r in self.robots]
        if len(positions) != len(set(positions)):
            violation = True
            violation_reasons.append("collision")

        # termination: all delivered or all dead
        all_delivered = all(r["load"] == 0 for r in self.robots)
        all_dead = all(r["battery"] <= 0.0 for r in self.robots)
        done = all_delivered or all_dead

        # -----------------------
        # Reward per PDF (strict)
        # rt = - sum_i ( a*(W_i)^2 * d_i + b ) - lambda_viol * 1{viol} + r_goal * 1{all delivered}
        # -----------------------
        r_energy = -float(sum(energy_costs))
        r_viol = -self.lambda_viol if violation else 0.0
        r_goal = float(self.r_goal) if all_delivered else 0.0
        reward = r_energy + r_viol + r_goal

        # optional small shaping (helps learning) — controlled by flag
        if self.shaped_reward:
            # dense guidance: reward for reducing Manhattan distance to destination
            dest_x, dest_y = self.grid_size - 1, self.grid_size - 1
            for i, (old, rnew) in enumerate(zip(old_states, self.robots)):
                old_x, old_y, old_load, _ = old
                new_x, new_y = rnew["x"], rnew["y"]
                old_dist = abs(dest_x - old_x) + abs(dest_y - old_y)
                new_dist = abs(dest_x - new_x) + abs(dest_y - new_y)
                delta = old_dist - new_dist
                reward += 0.5 * float(max(0.0, delta))
                if new_dist > old_dist:
                    reward -= 0.2

            # small coop reward for each transfer
            reward += 5.0 * len(transfers)

        # info dictionary
        info = {
            "energy_costs": energy_costs,
            "transfers": transfers,
            "deliveries": deliveries,
            "deliveries_total": sum(1 for r in self.robots if r["load"] == 0),
            "violation": violation,
            "violation_reasons": violation_reasons,
            "d_list": d_list,
            "r_energy": r_energy,
            "r_viol": r_viol,
            "r_goal": r_goal,
        }

        if self.debug:
            print(f"Step info: transfers={transfers}, deliveries={deliveries}, reward={reward:.2f}, done={done}")

        return self._get_obs(), float(reward), bool(done), False, info
