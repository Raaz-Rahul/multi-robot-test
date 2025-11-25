import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Movement deltas (N,S,E,W,Stay)
MOVE_DIRS = {
    0: (-1, 0),  # N
    1: (1, 0),   # S
    2: (0, 1),   # E
    3: (0, -1),  # W
    4: (0, 0),   # Stay
}


class MultiRobotEnv(gym.Env):
    """
    Centralized RL environment for energy-aware multi-robot delivery.

    Phases (per document):
    - Phase 1: multi-robot, movement only, team energy reward.
    - Phase 2: add energy forecast features to observation.
    - Phase 3: same env, but used to generate expert demos.
    - Phase 4: enable sharing heads (offload, partner, meeting, amount) with masks.
    - Phase 5: enforce hard constraints (battery, meeting feasibility, collision) via masks.
    """

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

        # Energy parameters e_i = a (W_i^2) d + b
        self.a = a_energy
        self.b = b_energy
        self.battery_budget = battery_budget
        self.max_load = max_load
        self.share_radius = share_radius
        self.beta = beta_threshold
        self.violation_penalty = violation_penalty
        self.goal_reward = goal_reward

        # Simple map: 2 storages + 1 destination
        self.destination = (self.H - 1, self.W - 1)
        self.storage_nodes = [(0, 0), (0, self.W - 1)]
        self.init_inventory = np.array([4, 4], dtype=np.int32)

        # State variables
        self.pos = None           # (n,2)
        self.load = None          # (n,)
        self.battery = None       # (n,)
        self.inventory = None     # (|S|,)
        self.forecast = None      # ̂E_i
        self.offload_indicator = None  # u_i
        self.routes = None        # per-robot remaining path to destination (list of lists)
        self.t = 0

        # Observation encoding (Section 7) [attached_file:1]
        # Per robot: (x_norm, y_norm, load_norm, battery_ratio, forecast_norm, offload_flag)
        robot_dim = 6
        global_dim = len(self.storage_nodes) + 2  # inventory + dest coords
        self.obs_dim = num_robots * robot_dim + global_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Factorized action space (Eq. 2) [attached_file:1]
        # Categorical heads per robot:
        #  m_i in {0..4}, o_i in {0,1}, j_i in {0..n-1}, M_i in {0..H*W-1}, ω_i in {0..max_load}
        self.num_move = 5
        self.num_offload = 2
        self.num_partner = self.num_robots
        self.num_meeting = self.H * self.W
        self.num_amount = self.max_load + 1

        # We use factorized heads (not packed into single MultiDiscrete int)
        # The training code will treat each head separately; here we accept a dict or concatenated.
        self.action_space = spaces.Dict(
            {
                "move": spaces.MultiDiscrete([self.num_move] * self.num_robots),
                "offload": spaces.MultiDiscrete([self.num_offload] * self.num_robots),
                "partner": spaces.MultiDiscrete([self.num_partner] * self.num_robots),
                "meeting": spaces.MultiDiscrete([self.num_meeting] * self.num_robots),
                "amount": spaces.MultiDiscrete([self.num_amount] * self.num_robots),
            }
        )

    # ------------- Core API ------------- #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # Initialize robots at random non-destination nodes
        self.pos = np.zeros((self.num_robots, 2), dtype=np.int32)
        for i in range(self.num_robots):
            self.pos[i] = self._sample_non_dest()

        self.load = np.zeros(self.num_robots, dtype=np.int32)
        self.battery = np.full(self.num_robots, self.battery_budget, dtype=np.float32)
        self.inventory = self.init_inventory.copy()
        self.routes = [self._shortest_path_to_dest(tuple(self.pos[i])) for i in range(self.num_robots)]

        self._update_forecasts_and_flags()
        obs = self._encode_obs()
        info = {"masks": self._compute_masks()}
        return obs, info

    def step(self, action):
        """
        action: dict with keys 'move', 'offload', 'partner', 'meeting', 'amount'
        Each value is array-like of length num_robots.
        """
        self.t += 1
        # Ensure arrays
        move = np.array(action["move"], dtype=np.int64)
        offload = np.array(action["offload"], dtype=np.int64)
        partner = np.array(action["partner"], dtype=np.int64)
        meeting = np.array(action["meeting"], dtype=np.int64)
        amount = np.array(action["amount"], dtype=np.int64)

        masks = self._compute_masks()  # use current state

        # Apply masks in phase 5: clamp to feasible by zeroing invalid actions
        # (policy should avoid them anyway due to masked sampling)
        violations = 0
        if self.phase >= 5:
            v1 = self._count_violations(move, offload, partner, meeting, amount, masks)
            violations += v1

        # 1) Movement and collision handling [attached_file:1]
        energy_step = self._apply_movement(move)

        # 2) Auto pick/drop on storage & destination [attached_file:1]
        self._auto_pick_drop()

        # 3) Sharing (phases 4–5) [attached_file:1]
        if self.phase >= 4:
            v2 = self._apply_sharing(offload, partner, meeting, amount, masks)
            violations += v2

        # 4) Energy accounting, battery update, reward (Eq. 3) [attached_file:1]
        self.battery -= energy_step
        base_cost = np.sum(energy_step)
        reward = -float(base_cost)
        if violations > 0:
            reward -= self.violation_penalty * violations

        success = self._all_delivered()
        if success:
            reward += self.goal_reward

        # 5) Termination [attached_file:1]
        done = False
        truncated = False
        if success:
            done = True
        if self.t >= self.max_steps:
            done = True
            truncated = True
        if np.any(self.battery < 0.0):
            done = True

        # 6) Update forecasts and masks (phases 2+) [attached_file:1]
        if self.phase >= 2:
            self._update_forecasts_and_flags()

        obs = self._encode_obs()
        info = {"masks": self._compute_masks(), "success": success}
        return obs, reward, done, truncated, info

    # ------------- Helpers: geometry & routes ------------- #

    def _sample_non_dest(self):
        while True:
            x = self.np_random.integers(0, self.H)
            y = self.np_random.integers(0, self.W)
            if (x, y) != self.destination:
                return np.array([x, y], dtype=np.int32)

    def _shortest_path_to_dest(self, start):
        # Simple Manhattan-path route to destination (no obstacles). [attached_file:1]
        path = []
        x, y = start
        dx, dy = self.destination
        while x != dx:
            step = (1 if dx > x else -1)
            x += step
            path.append((x, y))
        while y != dy:
            step = (1 if dy > y else -1)
            y += step
            path.append((x, y))
        return path

    def _manhattan(self, u, v):
        return abs(u[0] - v[0]) + abs(u[1] - v[1])

    # ------------- Movement & energy ------------- #

    def _apply_movement(self, move):
        # Proposed next positions
        next_pos = self.pos.copy()
        for i in range(self.num_robots):
            d = MOVE_DIRS[int(move[i])]
            nx = self.pos[i, 0] + d[0]
            ny = self.pos[i, 1] + d[1]
            if 0 <= nx < self.H and 0 <= ny < self.W:
                next_pos[i] = (nx, ny)
            else:
                next_pos[i] = self.pos[i]  # stay if out of grid

        # Collision: if two robots choose same node, keep them in place (simple resolution). [attached_file:1]
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if np.array_equal(next_pos[i], next_pos[j]):
                    next_pos[i] = self.pos[i]
                    next_pos[j] = self.pos[j]

        # Energy per robot using Eq. (1) with d=1 if moved, else 0. [attached_file:1]
        moved = np.any(next_pos != self.pos, axis=1).astype(np.float32)
        energy = self.a * (self.load.astype(np.float32) ** 2) * moved + self.b * moved

        self.pos = next_pos

        # Update routes: remove visited nodes from planned path
        for i in range(self.num_robots):
            path = self.routes[i]
            if len(path) > 0 and tuple(self.pos[i]) == path[0]:
                self.routes[i] = path[1:]

        return energy

    # ------------- Pick/drop & inventory ------------- #

    def _auto_pick_drop(self):
        # At storage: pick until either robot full or inventory empty.
        for r in range(self.num_robots):
            pr = tuple(self.pos[r])
            for idx, sn in enumerate(self.storage_nodes):
                if pr == sn and self.inventory[idx] > 0 and self.load[r] < self.max_load:
                    can_take = min(self.max_load - self.load[r], self.inventory[idx])
                    self.load[r] += can_take
                    self.inventory[idx] -= can_take

        # At destination: deliver all load.
        for r in range(self.num_robots):
            if tuple(self.pos[r]) == self.destination and self.load[r] > 0:
                self.load[r] = 0

    def _all_delivered(self):
        return np.all(self.inventory == 0) and np.all(self.load == 0)

    # ------------- Sharing logic (phases 4–5) ------------- #

    def _apply_sharing(self, offload, partner, meeting, amount, masks):
        violations = 0
        # We interpret sharing as instantaneous when robots are already at same meeting node.
        # The heuristic generator will choose partners and meeting nodes using radius constraints. [attached_file:1]

        # Build proposals
        proposals = []
        for i in range(self.num_robots):
            if offload[i] == 0:
                continue
            j = int(partner[i])
            if j == i or j < 0 or j >= self.num_robots:
                continue
            M_idx = int(meeting[i])
            M = (M_idx // self.W, M_idx % self.W)
            w = int(amount[i])
            proposals.append((i, j, M, w))

        # Apply valid proposals (one per pair max)
        used_pairs = set()
        for (i, j, M, w) in proposals:
            if (i, j) in used_pairs or (j, i) in used_pairs:
                continue
            # Feasibility masks in phase 5
            if self.phase >= 5:
                # Partner mask
                if masks["partner"][i, j] == 0:
                    violations += 1
                    continue
                # Meeting mask: M must be feasible for i
                M_idx = M[0] * self.W + M[1]
                if masks["meeting"][i, M_idx] == 0:
                    violations += 1
                    continue

            # Simple implementation: sharing only if both robots are actually at M now.
            if tuple(self.pos[i]) != M or tuple(self.pos[j]) != M:
                continue

            # Capacity / load constraints (Section 2.5) [attached_file:1]
            w = max(0, min(w, self.load[i]))
            cap_j = self.max_load - self.load[j]
            w = min(w, cap_j)
            if w <= 0:
                continue

            self.load[i] -= w
            self.load[j] += w
            used_pairs.add((i, j))

        return violations

    # ------------- Forecast and offload indicator (Section 3) ------------- #

    def _update_forecasts_and_flags(self):
        self.forecast = np.zeros(self.num_robots, dtype=np.float32)
        for i in range(self.num_robots):
            path = self.routes[i]
            if len(path) == 0:
                # Already at destination; forecast is 0
                self.forecast[i] = 0.0
                continue
            # Simple estimate: assume current load along remaining path with d=1 steps. [attached_file:1]
            L = len(path)
            self.forecast[i] = np.sum(
                self.a * (self.load[i] ** 2) * 1.0 + self.b
                for _ in range(L)
            )

        # Offload indicator u_i = 1{ ̂E_i > beta * B_i } (Eq. 4) [attached_file:1]
        self.offload_indicator = (self.forecast > self.beta * self.battery_budget).astype(np.float32)

    # ------------- Masks (Section 2.5 & 7) ------------- #

    def _compute_masks(self):
        """
        Returns a dict of masks per head:
        - movement: [n, num_move]
        - partner: [n, num_partner]
        - meeting: [n, num_meeting]
        - amount: [n, num_amount]
        Used directly by the policy to restrict sampling. [attached_file:1]
        """
        n = self.num_robots
        movement = np.ones((n, self.num_move), dtype=np.float32)
        partner = np.ones((n, self.num_partner), dtype=np.float32)
        meeting = np.ones((n, self.num_meeting), dtype=np.float32)
        amount = np.ones((n, self.num_amount), dtype=np.float32)

        # Early phases: disable irrelevant heads
        if self.phase < 4:
            partner[:] = 0.0
            meeting[:] = 0.0
            amount[:] = 0.0

        # Battery feasibility (mask moves whose projected forecast would exceed remaining budget). [attached_file:1]
        for i in range(n):
            for m in range(self.num_move):
                # approximate: if moving one step with current load would exceed remaining battery, mask.
                moved = 0.0 if m == 4 else 1.0
                e_step = self.a * (self.load[i] ** 2) * moved + self.b * moved
                if self.battery[i] - e_step < 0.0:
                    movement[i, m] = 0.0

        # Partner feasibility: only allow within distance <= 2r and j != i. [attached_file:1]
        for i in range(n):
            for j in range(n):
                if i == j:
                    partner[i, j] = 0.0
                else:
                    if self._manhattan(self.pos[i], self.pos[j]) > 2 * self.share_radius:
                        partner[i, j] = 0.0

        # Meeting node feasibility: Δ(p_i, M) <= r and optionally close to some partner. [attached_file:1]
        for i in range(n):
            for idx in range(self.num_meeting):
                mx = idx // self.W
                my = idx % self.W
                if self._manhattan(self.pos[i], (mx, my)) > self.share_radius:
                    meeting[i, idx] = 0.0

        # Amount mask: 0 <= ω <= W_i and <= capacity of some partner.
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
        # Count actions that pick masked choices
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

    # ------------- Observation encoding (Section 7) ------------- #

    def _encode_obs(self):
        feats = []

        def norm_coord(x, maxv):
            return 2.0 * (x / (maxv - 1 + 1e-8)) - 1.0

        for i in range(self.num_robots):
            x, y = self.pos[i]
            x_n = norm_coord(x, self.H)
            y_n = norm_coord(y, self.W)
            load_n = self.load[i] / float(self.max_load + 1e-8)
            batt_ratio = self.battery[i] / (self.battery_budget + 1e-8)
            if self.phase >= 2:
                forecast_n = self.forecast[i] / (self.battery_budget * 4.0 + 1e-8)
                u_i = self.offload_indicator[i]
            else:
                forecast_n = 0.0
                u_i = 0.0
            feats.extend([x_n, y_n, load_n, batt_ratio, forecast_n, u_i])

        # Global: inventory, destination coords. [attached_file:1]
        for k in range(len(self.storage_nodes)):
            feats.append(self.inventory[k] / float(self.init_inventory[k] + 1e-8))
        dx_n = norm_coord(self.destination[0], self.H)
        dy_n = norm_coord(self.destination[1], self.W)
        feats.extend([dx_n, dy_n])

        return np.array(feats, dtype=np.float32)

    # ------------- Render ------------- #

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
        print(f"t={self.t}")
        for row in grid:
            print(" ".join(row))
        print("load:", self.load, "battery:", self.battery)
