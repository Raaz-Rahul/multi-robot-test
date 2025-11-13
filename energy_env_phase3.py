import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnvPhase3(gym.Env):
    """
    Cooperative multi-robot environment (Phase 3) with denser reward shaping.
    Actions per robot:
      0 = stay
      1 = north (y+1)
      2 = south (y-1)
      3 = east  (x+1)
      4 = west  (x-1)
      5 = request transfer (ask nearby idle robot to take your load)
    """
    metadata = {"render_modes": []}

    def __init__(self, n_robots=3, grid_size=5, battery_init=50, debug=False):
        super().__init__()

        self.n_robots = n_robots
        self.grid_size = grid_size
        self.debug = debug

        # allow action 0..5 (6 discrete choices)
        self.action_space = spaces.MultiDiscrete([6] * n_robots)

        # observation: for each robot -> [x, y, load, battery]
        high = np.array([grid_size, grid_size, 1, battery_init] * n_robots, dtype=float)
        self.observation_space = spaces.Box(0.0, high, dtype=float)

        # energy model
        self.battery_init = float(battery_init)
        self.a = 0.1
        self.b = 0.1

        # rewards / penalties
        self.lambda_violation = 10.0   # penalty for collision or battery < 0
        self.coop_transfer_reward = 5.0
        self.coop_bonus = 20.0         # legacy coop bonus (if you want larger)
        self.delivery_reward = 100.0
        self.fail_episode_penalty = 10.0

        # cooperation params
        self.low_battery_threshold = 10.0
        self.comm_range = 2  # manhattan

        # runtime state
        self.robots = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robots = []
        for i in range(self.n_robots):
            self.robots.append({
                "x": i,
                "y": 0,
                "load": 1,                     # 1 => carrying parcel; 0 => no parcel
                "battery": float(self.battery_init),
                "destination": (self.grid_size - 1, self.grid_size - 1)
            })

        # no global step counter needed here but can be added
        return self._get_obs(), {}

    # -----------------------------------------------------
    def step(self, actions):
        """
        actions: iterable/list of length n_robots with integers in [0..5]
        returns: obs, reward, done, trunc(False), info(dict)
        """

        assert len(actions) == self.n_robots, "Action vector length mismatch"

        # Record old states for dense reward calculations (before movement)
        old_states = []
        for r in self.robots:
            old_states.append((r["x"], r["y"], r["load"], r["battery"]))

        # ------------------ Movement (apply actions) ------------------
        for i, act in enumerate(actions):
            r = self.robots[i]
            x, y = r["x"], r["y"]

            # movement actions
            if act == 1 and y < self.grid_size - 1:
                y += 1
            elif act == 2 and y > 0:
                y -= 1
            elif act == 3 and x < self.grid_size - 1:
                x += 1
            elif act == 4 and x > 0:
                x -= 1
            # act == 0 => stay
            # act == 5 => request transfer (handled below)

            r["x"], r["y"] = x, y

        # ------------------ Energy cost (after movement) ------------------
        energy_costs = []
        for i in range(self.n_robots):
            load = self.robots[i]["load"]
            d = 1.0   # assume 1 unit distance per step when movement occurs (simplified)
            cost = self.a * (load ** 2) * d + self.b
            self.robots[i]["battery"] -= cost
            energy_costs.append(cost)

        # ------------------ Cooperative transfers ------------------
        # We'll process explicit transfer requests (action==5) first, then automatic helper logic.
        transfers = []   # list of (from_idx, to_idx)
        for i, act in enumerate(actions):
            if act == 5:
                # requester i asks for helper: find nearest robot with load==0 and enough battery
                requester = self.robots[i]
                # only if requester still has load
                if requester["load"] == 1:
                    nearest = None
                    nearest_dist = 1e9
                    for j, helper in enumerate(self.robots):
                        if j == i: continue
                        if helper["load"] == 0 and helper["battery"] > self.low_battery_threshold:
                            d = abs(requester["x"] - helper["x"]) + abs(requester["y"] - helper["y"])
                            if d <= self.comm_range and d < nearest_dist:
                                nearest = j
                                nearest_dist = d
                    if nearest is not None:
                        # perform transfer
                        self.robots[i]["load"] = 0
                        self.robots[nearest]["load"] = 1
                        transfers.append((i, nearest))

        # Also automatic transfer: low-battery robots may transfer to idle nearby robots
        for i in range(self.n_robots):
            r_i = self.robots[i]
            if r_i["load"] == 1 and r_i["battery"] < self.low_battery_threshold:
                for j in range(self.n_robots):
                    if i == j: continue
                    r_j = self.robots[j]
                    if r_j["load"] == 0 and r_j["battery"] > self.low_battery_threshold:
                        d = abs(r_i["x"] - r_j["x"]) + abs(r_i["y"] - r_j["y"])
                        if d <= self.comm_range:
                            r_i["load"] = 0
                            r_j["load"] = 1
                            transfers.append((i, j))
                            # break to avoid multiple transfers for same robot this step
                            break

        # ------------------ Delivery checks ------------------
        deliveries = []
        for i in range(self.n_robots):
            r = self.robots[i]
            if r["load"] == 1:
                dx, dy = r["destination"]
                if (r["x"], r["y"]) == (dx, dy):
                    # deliver
                    r["load"] = 0
                    deliveries.append(i)

        # ------------------ Violation checks (battery <0 or collisions) ------------------
        violation = False
        violation_penalty = 0.0

        for i in range(self.n_robots):
            if self.robots[i]["battery"] < 0.0:
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

        # ------------------ Termination check ------------------
        all_delivered = all(r["load"] == 0 for r in self.robots)
        all_dead = all(r["battery"] <= 0.0 for r in self.robots)
        done = all_delivered or all_dead

        # ------------------ Dense reward assembly ------------------
        # Base: negative energy cost
        reward = -sum(energy_costs)

        # Dense guiding reward: change in Manhattan distance to the (shared) destination
        dest_x, dest_y = self.grid_size - 1, self.grid_size - 1

        # new states for comparison
        new_states = []
        for r in self.robots:
            new_states.append((r["x"], r["y"], r["load"], r["battery"]))

        # reward for moving closer, small penalty for moving away
        for old, new in zip(old_states, new_states):
            old_x, old_y, old_load, _ = old
            new_x, new_y, new_load, _ = new
            old_dist = abs(old_x - dest_x) + abs(old_y - dest_y)
            new_dist = abs(new_x - dest_x) + abs(new_y - dest_y)
            if new_dist < old_dist:
                reward += 0.5 * (old_dist - new_dist)
            elif new_dist > old_dist:
                reward -= 0.2 * (new_dist - old_dist)

        # reward for each successful transfer this step (small positive)
        if len(transfers) > 0:
            reward += self.coop_transfer_reward * len(transfers)

        # bigger bonus for delivery events this step
        if len(deliveries) > 0:
            reward += self.delivery_reward * len(deliveries)

        # penalty if episode ends and no deliveries happened in total
        if done and not all_delivered:
            reward -= self.fail_episode_penalty

        # include violation penalty (already negative)
        reward += violation_penalty

        # ------------------ info dictionary ------------------
        deliveries_total = sum(1 for r in self.robots if r["load"] == 0)
        info = {
            "transfers": transfers,              # list of (from, to)
            "deliveries": deliveries,            # list of robots delivered this step
            "deliveries_total": deliveries_total,
            "violation": violation
        }

        if self.debug:
            print(f"Step info: transfers={transfers}, deliveries={deliveries}, reward={reward:.2f}, done={done}")

        return self._get_obs(), float(reward), bool(done), False, info

    # -----------------------------------------------------
    def _get_obs(self):
        flat = []
        for r in self.robots:
            flat.extend([float(r["x"]), float(r["y"]), float(r["load"]), float(r["battery"])])
        return np.array(flat, dtype=float)
