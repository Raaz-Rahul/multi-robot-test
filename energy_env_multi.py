import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnv(gym.Env):
    """
    Phase 1: Centralized multi-robot (no load sharing, constraint penalties)

    - N robots on a grid.
    - Each robot starts with 1 item to deliver.
    - Joint action: each robot chooses from {0=Stay, 1=N, 2=S, 3=E, 4=W}.
    - Reward = -(sum of energy costs) - λ_viol*(constraint violation) + r_goal*(all delivered).
    - Constraints handled:
        * Battery feasibility: robot battery < 0
        * Collision: two robots in the same cell
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_robots=3, grid_size=10, battery=100, a=1.0, b=0.1,
                 r_goal=50.0, lambda_viol=10.0, debug=False):
        super().__init__()
        assert n_robots >= 1
        self.n = n_robots
        self.grid_size = grid_size
        self.a, self.b = a, b
        self.battery_init = battery
        self.r_goal = float(r_goal)
        self.lambda_viol = float(lambda_viol)
        self.debug = debug  # ✅ new flag

        # State per robot = [x, y, load, battery]
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(grid_size, battery),
            shape=(4 * self.n,),
            dtype=np.float32
        )

        # Joint action: one discrete action per robot
        self.action_space = spaces.MultiDiscrete([5] * self.n)

        # Storage (pickup) and destination (drop-off)
        self.storage = (0, 0)
        self.destination = (grid_size - 1, grid_size - 1)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start robots along x-axis of row 0
        self.pos = np.zeros((self.n, 2), dtype=np.int32)
        for i in range(self.n):
            self.pos[i] = np.array([min(i, self.grid_size - 1), 0], dtype=np.int32)

        self.load = np.ones(self.n, dtype=np.int32)       # each robot carries 1 item
        self.battery = np.full(self.n, self.battery_init, dtype=np.float32)
        self.done = False

        return self._get_state(), {}

    def step(self, actions):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.n,)

        total_energy = 0.0
        violated = False
        violation_reasons = []  # ✅ track violations

        # Apply actions for each robot
        for i in range(self.n):
            a = actions[i]
            moved = 0

            # Move with boundary checks
            if a == 1 and self.pos[i, 1] < self.grid_size - 1:      # North
                self.pos[i, 1] += 1; moved = 1
            elif a == 2 and self.pos[i, 1] > 0:                     # South
                self.pos[i, 1] -= 1; moved = 1
            elif a == 3 and self.pos[i, 0] < self.grid_size - 1:    # East
                self.pos[i, 0] += 1; moved = 1
            elif a == 4 and self.pos[i, 0] > 0:                     # West
                self.pos[i, 0] -= 1; moved = 1
            # else Stay (0) or out-of-bounds → no movement

            # Energy cost
            W = float(self.load[i])
            e = self.a * (W ** 2) * moved + self.b
            self.battery[i] -= e
            total_energy += e

            # Battery violation
            if self.battery[i] < 0:
                violated = True
                violation_reasons.append(f"R{i} battery < 0")

        # Auto pickup/drop (no sharing)
        for i in range(self.n):
            if tuple(self.pos[i]) == self.storage and self.load[i] == 0:
                self.load[i] = 1
            if tuple(self.pos[i]) == self.destination and self.load[i] > 0:
                self.load[i] = 0

        # Collision violation if two or more robots share a cell
        unique_positions = {tuple(p) for p in self.pos}
        if len(unique_positions) < self.n:
            violated = True
            violation_reasons.append("Collision")

        # Termination
        all_delivered = bool(np.all(self.load == 0))
        all_dead = bool(np.all(self.battery <= 0.0))
        terminated = all_delivered or all_dead
        truncated = False

        # Reward
        reward = -total_energy
        if violated:
            reward -= self.lambda_viol
            if self.debug:  # ✅ only print if debug enabled
                print("Constraint violated:", ", ".join(violation_reasons))
        if all_delivered:
            reward += self.r_goal

        self.done = terminated
        return self._get_state(), reward, terminated, truncated, {}

    def _get_state(self):
        per_robot = []
        for i in range(self.n):
            per_robot.extend([
                float(self.pos[i, 0]),
                float(self.pos[i, 1]),
                float(self.load[i]),
                float(self.battery[i])
            ])
        return np.array(per_robot, dtype=np.float32)

    def render(self):
        rows = []
        for i in range(self.n):
            rows.append(
                f"R{i}: [{int(self.pos[i,0])} {int(self.pos[i,1])} {int(self.load[i])} {round(float(self.battery[i]),2)}]"
            )
        print(" | ".join(rows))
