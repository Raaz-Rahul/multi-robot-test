import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SingleRobotEnergyEnv(gym.Env):
    """
    Phase 0 Base Environment:
    - 1 robot on a grid
    - Picks up item from storage
    - Delivers to destination
    - Reward = -energy consumption + goal bonus
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=10, battery=100, a=1.0, b=0.1):
        super(SingleRobotEnergyEnv, self).__init__()

        self.grid_size = grid_size
        self.a, self.b = a, b
        self.battery_init = battery

        # State = [robot_x, robot_y, load, battery_remaining]
        self.observation_space = spaces.Box(
            low=0,
            high=max(grid_size, battery),
            shape=(4,),
            dtype=np.float32
        )

        # Actions = {0:Stay, 1:North, 2:South, 3:East, 4:West}
        self.action_space = spaces.Discrete(5)

        # Storage (pickup) & Destination (dropoff)
        self.storage = (0, 0)
        self.destination = (grid_size - 1, grid_size - 1)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0])      # robot starts at (0,0)
        self.load = 1                    # 1 item to deliver
        self.battery = self.battery_init
        self.done = False
        return self._get_state(), {}

    def _get_state(self):
        return np.array([self.pos[0], self.pos[1], self.load, self.battery], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Movement
        if action == 1 and self.pos[1] < self.grid_size - 1:   # North
            self.pos[1] += 1
        elif action == 2 and self.pos[1] > 0:                  # South
            self.pos[1] -= 1
        elif action == 3 and self.pos[0] < self.grid_size - 1: # East
            self.pos[0] += 1
        elif action == 4 and self.pos[0] > 0:                  # West
            self.pos[0] -= 1
        # action==0 means Stay (no movement)

        # Energy consumption
        distance = 1 if action != 0 else 0
        energy_used = self.a * (self.load ** 2) * distance + self.b
        self.battery -= energy_used

        reward = -energy_used
        terminated = False
        truncated = False

        # Automatic pickup/drop
        if tuple(self.pos) == self.storage and self.load == 0:
            self.load = 1
        if tuple(self.pos) == self.destination and self.load > 0:
            self.load = 0
            reward += 50   # bonus for task completion
            terminated = True

        # End if battery drained
        if self.battery <= 0:
            terminated = True

        self.done = terminated
        return self._get_state(), reward, terminated, truncated, {}

    def render(self):
        print(f"State: [{int(self.pos[0])} {int(self.pos[1])} {int(self.load)} {round(self.battery,2)}]")
