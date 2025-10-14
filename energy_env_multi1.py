import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiRobotEnergyEnv(gym.Env):
    """
    Centralized Multi-Robot Environment for Pickup and Delivery Tasks
    ---------------------------------------------------------------
    - Each robot moves on a 2D grid.
    - Each starts with one parcel to deliver.
    - Reward encourages energy efficiency and delivery success.
    - Constraints: battery, collisions, and boundary limits.
    """

    def __init__(self, n_robots=3, grid_size=5, battery=50, debug=False):
        super(MultiRobotEnergyEnv, self).__init__()

        self.num_robots = n_robots
        self.grid_size = grid_size
        self.battery_init = battery
        self.debug = debug

        # --- Define action and observation spaces ---
        # 5 actions: [Stay, North, South, East, West]
        self.action_space = spaces.MultiDiscrete([5] * self.num_robots)

        # Observation: for each robot → [x, y, load, battery]
        low = np.array([0, 0, 0, 0] * self.num_robots, dtype=np.float32)
        high = np.array(
            [grid_size - 1, grid_size - 1, 1, battery] * self.num_robots, dtype=np.float32
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # --- Parameters for reward ---
        self.a = 0.1     # energy coefficient (load^2 * distance)
        self.b = 0.1     # base cost
        self.lambda_viol = 10  # violation penalty
        self.r_goal = 100      # delivery reward bonus

        self.reset()

    # ---------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        # Initial robot states
        self.positions = np.array([[i, 0] for i in range(self.num_robots)], dtype=int)
        self.batteries = np.ones(self.num_robots) * self.battery_init
        self.loads = np.ones(self.num_robots)  # 1 = carrying parcel

        # Define pickup and delivery points
        self.pickup_positions = [(0, 0) for _ in range(self.num_robots)]
        self.delivery_positions = [(self.grid_size - 1, self.grid_size - 1) for _ in range(self.num_robots)]

        obs = self._get_obs()
        info = {}
        return obs, info

    # ---------------------------------------------------
    def _get_obs(self):
        obs = []
        for i in range(self.num_robots):
            x, y = self.positions[i]
            obs.extend([x, y, self.loads[i], self.batteries[i]])
        return np.array(obs, dtype=np.float32)

    # ---------------------------------------------------
    def step(self, actions):
        reward = 0.0
        terminated = False
        violations = []

        actions = np.array(actions, dtype=int)

        # --- Move each robot based on action ---
        for i in range(self.num_robots):
            if self.batteries[i] <= 0:
                continue  # skip dead robot

            x, y = self.positions[i]
            a = actions[i]

            if a == 1 and y < self.grid_size - 1:  # North
                y += 1
            elif a == 2 and y > 0:  # South
                y -= 1
            elif a == 3 and x < self.grid_size - 1:  # East
                x += 1
            elif a == 4 and x > 0:  # West
                x -= 1
            # a == 0 → stay

            self.positions[i] = [x, y]

        # --- Check for collisions ---
        unique_positions = [tuple(p) for p in self.positions]
        if len(unique_positions) > len(set(unique_positions)):
            violations.append("Collision")

        # --- Compute energy consumption and battery update ---
        for i in range(self.num_robots):
            distance_cost = self.a * (self.loads[i] ** 2) + self.b
            self.batteries[i] -= distance_cost
            reward -= distance_cost

        # --- Check for battery violations ---
        for i in range(self.num_robots):
            if self.batteries[i] < 0:
                violations.append(f"R{i} battery < 0")

        # --- Pickup and delivery logic ---
        for i in range(self.num_robots):
            px, py = self.pickup_positions[i]
            dx, dy = self.delivery_positions[i]
            cx, cy = self.positions[i]

            # Pickup: robot at pickup location, not carrying
            if (cx, cy) == (px, py) and self.loads[i] == 0:
                self.loads[i] = 1
                if self.debug:
                    print(f"R{i} picked up parcel.")

            # Delivery: robot at delivery location, carrying parcel
            if (cx, cy) == (dx, dy) and self.loads[i] == 1:
                self.loads[i] = 0
                reward += self.r_goal
                if self.debug:
                    print(f"R{i} delivered parcel! +{self.r_goal}")

        # --- Violation penalties ---
        if violations:
            reward -= self.lambda_viol
            if self.debug:
                print("Constraint violated:", ", ".join(violations))

        # --- Check for episode termination ---
        all_delivered = np.all(self.loads == 0)
        all_dead = np.all(self.batteries <= 0)
        if all_delivered or all_dead:
            terminated = True

        obs = self._get_obs()
        info = {"violations": violations}
        return obs, reward, terminated, False, info

    # ---------------------------------------------------
    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        for i, pos in enumerate(self.positions):
            x, y = pos
            grid[self.grid_size - 1 - y, x] = str(i)

        print("\nGrid:")
        for row in grid:
            print(" ".join(row))
        print("-" * 20)

    # ---------------------------------------------------
    def close(self):
        pass
