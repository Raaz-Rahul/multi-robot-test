import numpy as np
import gymnasium as gym
from gymnasium import spaces

MOVE_DIRS = {
    0: (-1, 0),  # N
    1: (1, 0),   # S
    2: (0, 1),   # E
    3: (0, -1),  # W
    4: (0, 0),   # Stay
}

class MultiRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        grid_size=(10, 10),
        num_robots=3,
        phase=1,
        max_steps=200,
        a_energy=1.0,
        b_energy=0.1,
        battery_budget=200.0,
        max_load=6,
        share_radius=5,
        beta_threshold=0.8,
        violation_penalty=10.0,
        goal_reward=100.0,
    ):
        super().__init__()
        self.H, self.W = grid_size
        self.num_robots = num_robots
        self.phase = phase
        self.max_steps = max_steps

        self.a = a_energy
        self.b = b_energy
        self.battery_budget = battery_budget
        self.max_load = max_load
        self.share_radius = share_radius
        self.beta = beta_threshold
        self.violation_penalty = violation_penalty
        self.goal_reward = goal_reward

        self.destination = (self.H - 1, self.W - 1)
        self.storage_nodes = [(0, 0), (0, self.W - 1)]
        self.init_inventory = np.array([4, 4], dtype=np.int32)

        self.np_random = np.random.default_rng()

        self.pos = None
        self.load = None
        self.battery = None
        self.inventory = None
        self.forecast = None
        self.offload_indicator = None
        self.routes = None
        self.t = 0

        self.collision_count = 0
        self.delivered_count = 0
        self.load_sharing_events = []
        self.battery_history = [[] for _ in range(self.num_robots)]
        self.load_history = [[] for _ in range(self.num_robots)]

        robot_dim = 6
        global_dim = len(self.storage_nodes) + 2
        self.obs_dim = num_robots * robot_dim + global_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.num_move = 5
        self.num_offload = 2
        self.num_partner = self.num_robots
        self.num_meeting = self.H * self.W
        self.num_amount = self.max_load + 1

        self.action_space = spaces.Dict(
            {
                "move": spaces.MultiDiscrete([self.num_move] * self.num_robots),
                "offload": spaces.MultiDiscrete([self.num_offload] * self.num_robots),
                "partner": spaces.MultiDiscrete([self.num_partner] * self.num_robots),
                "meeting": spaces.MultiDiscrete([self.num_meeting] * self.num_robots),
                "amount": spaces.MultiDiscrete([self.num_amount] * self.num_robots),
            }
        )

    def _sample_non_dest(self):
        while True:
            x = self.np_random.integers(0, self.H)
            y = self.np_random.integers(0, self.W)
            if (x, y) != self.destination:
                return np.array([x, y], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = np.zeros((self.num_robots, 2), dtype=np.int32)
        for i in range(self.num_robots):
            self.pos[i] = self._sample_non_dest()
        self.load = np.zeros(self.num_robots, dtype=np.int32)
        self.battery = np.full(self.num_robots, self.battery_budget, dtype=np.float32)
        self.inventory = self.init_inventory.copy()
        self.routes = [self._shortest_path_to_dest(tuple(self.pos[i])) for i in range(self.num_robots)]

        self.collision_count = 0
        self.delivered_count = 0
        self.load_sharing_events.clear()
        self.battery_history = [[] for _ in range(self.num_robots)]
        self.load_history = [[] for _ in range(self.num_robots)]

        self._update_forecasts_and_flags()
        obs = self._encode_obs()
        info = {"masks": self._compute_masks()}
        return obs, info

    def _shortest_path_to_dest(self, start):
        path = []
        x, y = start
        dx, dy = self.destination
        while x != dx:
            step = 1 if dx > x else -1
            x += step
            path.append((x, y))
        while y != dy:
            step = 1 if dy > y else -1
            y += step
            path.append((x, y))
        return path

    def step(self, action):
        self.t += 1
        move = np.array(action["move"], dtype=np.int64)
        offload = np.array(action["offload"], dtype=np.int64)
        partner = np.array(action["partner"], dtype=np.int64)
        meeting = np.array(action["meeting"], dtype=np.int64)
        amount = np.array(action["amount"], dtype=np.int64)

        masks = self._compute_masks()

        violations = 0
        if self.phase >= 5:
            violations += self._count_violations(move, offload, partner, meeting, amount, masks)

        energy_step = self._apply_movement(move)
        self._auto_pick_drop()

        if self.phase >= 4:
            violations += self._apply_sharing(offload, partner, meeting, amount, masks)

        self.battery -= energy_step
        base_cost = np.sum(energy_step)
        reward = -float(base_cost)
        if violations > 0:
            reward -= self.violation_penalty * violations

        success = self._all_delivered()
        if success:
            reward += self.goal_reward

        done = False
        truncated = False
        if success:
            done = True
        if self.t >= self.max_steps:
            done = True
            truncated = True
        if np.any(self.battery < 0.0):
            done = True

        for i in range(self.num_robots):
            self.battery_history[i].append(self.battery[i])
            self.load_history[i].append(self.load[i])

        if np.all(self.inventory == 0) and np.all(self.load == 0):
            self.delivered_count += 1

        if self.phase >= 2:
            self._update_forecasts_and_flags()

        obs = self._encode_obs()
        info = {"masks": self._compute_masks(), "success": success}
        return obs, reward, done, truncated, info

    def _apply_movement(self, move):
        next_pos = self.pos.copy()
        collisions_this_step = 0
        for i in range(self.num_robots):
            d = MOVE_DIRS[int(move[i])]
            nx = self.pos[i, 0] + d[0]
            ny = self.pos[i, 1] + d[1]
            if 0 <= nx < self.H and 0 <= ny < self.W:
                next_pos[i] = (nx, ny)
            else:
                next_pos[i] = self.pos[i]

        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if np.array_equal(next_pos[i], next_pos[j]):
                    collisions_this_step += 1
                    next_pos[i] = self.pos[i]
                    next_pos[j] = self.pos[j]

        self.collision_count += collisions_this_step
        self.pos = next_pos

        for i in range(self.num_robots):
            path = self.routes[i]
            if len(path) > 0 and tuple(self.pos[i]) == path[0]:
                self.routes[i] = path[1:]

        moved = np.any(next_pos != self.pos, axis=1).astype(np.float32)
        energy = self.a * (self.load.astype(np.float32) ** 2) * moved + self.b * moved
        return energy

    def _apply_sharing(self, offload, partner, meeting, amount, masks):
        violations = 0
        used_pairs = set()
        for i in range(self.num_robots):
            if offload[i] == 0:
                continue
            j = int(partner[i])
            if j == i or j < 0 or j >= self.num_robots:
                continue
            M_idx = int(meeting[i])
            M = (M_idx // self.W, M_idx % self.W)
            w = int(amount[i])
            if (i, j) in used_pairs or (j, i) in used_pairs:
                continue
            if self.phase >= 5:
                if masks["partner"][i, j] == 0:
                    violations += 1
                    continue
                if masks["meeting"][i, M_idx] == 0:
                    violations += 1
                    continue
            if tuple(self.pos[i]) != M or tuple(self.pos[j]) != M:
                continue
            w = max(0, min(w, self.load[i]))
            cap_j = self.max_load - self.load[j]
            w = min(w, cap_j)
            if w <= 0:
                continue
            self.load[i] -= w
            self.load[j] += w
            self.load_sharing_events.append((self.t, i, j, w))
            used_pairs.add((i, j))
        return violations

    def _auto_pick_drop(self):
        for r in range(self.num_robots):
            pr = tuple(self.pos[r])
            for idx, sn in enumerate(self.storage_nodes):
                if pr == sn and self.inventory[idx] > 0 and self.load[r] < self.max_load:
                    can_take = min(self.max_load - self.load[r], self.inventory[idx])
                    self.load[r] += can_take
                    self.inventory[idx] -= can_take
        for r in range(self.num_robots):
            if tuple(self.pos[r]) == self.destination and self.load[r] > 0:
                self.load[r] = 0

    def _all_delivered(self):
        return np.all(self.inventory == 0) and np.all(self.load == 0)

    def _update_forecasts_and_flags(self):
        self.forecast = np.zeros(self.num_robots, dtype=np.float32)
        for i in range(self.num_robots):
            path = self.routes[i]
            if len(path) == 0:
                self.forecast[i] = 0.0
                continue
            L = len(path)
            self.forecast[i] = np.sum(
                self.a * (self.load[i] ** 2) * 1.0 + self.b
                for _ in range(L)
            )
        self.offload_indicator = (self.forecast > self.beta * self.battery_budget).astype(np.float32)

    def _compute_masks(self):
        n = self.num_robots
        movement = np.ones((n, self.num_move), dtype=np.float32)
        partner = np.ones((n, self.num_partner), dtype=np.float32)
        meeting = np.ones((n, self.num_meeting), dtype=np.float32)
        amount = np.ones((n, self.num_amount), dtype=np.float32)

        if self.phase < 4:
            partner[:] = 0.0
            meeting[:] = 0.0
            amount[:] = 0.0

        for i in range(n):
            for m in range(self.num_move):
                moved = 0.0 if m == 4 else 1.0
                e_step = self.a * (self.load[i] ** 2) * moved + self.b * moved
                if self.battery[i] - e_step < 0.0:
                    movement[i, m] = 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    partner[i, j] = 0.0
                else:
                    if abs(self.pos[i, 0] - self.pos[j, 0]) + abs(self.pos[i, 1] - self.pos[j, 1]) > 2 * self.share_radius:
                        partner[i, j] = 0.0

        for i in range(n):
            for idx in range(self.num_meeting):
                mx = idx // self.W
                my = idx % self.W
                if abs(self.pos[i, 0] - mx) + abs(self.pos[i, 1] - my) > self.share_radius:
                    meeting[i, idx] = 0.0

        for i in range(n):
            for w in range(self.num_amount):
                if w > self.load[i]:
                    amount[i, w] = 0.0

        return {
            "movement": movement,
            "partner": partner,
            "meeting": meeting,
            "amount": amount,
        }

    def _count_violations(self, move, offload, partner, meeting, amount, masks):
        v = 0
        n = self.num_robots
        for i in range(n):
            if masks["movement"][i, move[i]] == 0:
                v += 1
            if self.phase >= 4:
                if offload[i] == 1:
                    if masks["partner"][i, partner[i]] == 0:
                        v += 1
                    if masks["meeting"][i, meeting[i]] == 0:
                        v += 1
                    if masks["amount"][i, amount[i]] == 0:
                        v += 1
        return v

    def _encode_obs(self):
        feats = []
        def norm_coord(x, maxv):
            return 2.0 * (x / (maxv - 1 + 1e-8)) - 1.0

        for i in range(self.num_robots):
            x, y = self.pos[i]
            feats.extend([
                norm_coord(x, self.H),
                norm_coord(y, self.W),
                self.load[i] / (self.max_load + 1e-8),
                self.battery[i] / (self.battery_budget + 1e-8),
                self.forecast[i] / (self.battery_budget * 4.0 + 1e-8) if self.phase >= 2 else 0,
                self.offload_indicator[i] if self.phase >= 2 else 0.0,
            ])
        for k in range(len(self.storage_nodes)):
            feats.append(self.inventory[k] / (self.init_inventory[k] + 1e-8))
        feats.append(norm_coord(self.destination[0], self.H))
        feats.append(norm_coord(self.destination[1], self.W))
        return np.array(feats, dtype=np.float32)

    def render(self):
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        dx, dy = self.destination
        grid[dx][dy] = "D"
        for idx, s in enumerate(self.storage_nodes):
            x, y = s
            grid[x][y] = "S"
        for i in range(self.num_robots):
            x, y = self.pos[i]
            grid[x][y] = str(i)
        print(f"Step {self.t}")
        for row in grid:
            print(" ".join(row))
        print(f"Load: {self.load}, Battery: {self.battery}")
