# env.py
# Multi-robot load-sharing environment (Phases 0–5)
# Destination is sampled uniformly at reset.
# Reference spec (local): /mnt/data/RL Energy Shared Multi-Robot.pdf

import numpy as np
from typing import Dict, Tuple, Optional, Any

REFERENCE_PDF = "/mnt/data/RL Energy Shared Multi-Robot.pdf"

class MultiRobotEnv:
    """
    Centralized multi-robot delivery environment.
    Per-robot action block: [move, offload, partner, meeting_idx, w_idx]
    move: 0 stay, 1 up, 2 down, 3 left, 4 right
    """

    def __init__(self,
                 grid_size: Tuple[int,int] = (10, 10),
                 n_robots: int = 3,
                 storage_nodes: Optional[Dict[Tuple[int,int],int]] = None,
                 dest: Optional[Tuple[int,int]] = None,
                 init_battery: float = 100.0,
                 a: float = 1.0,
                 b: float = 0.1,
                 share_radius: int = 5,
                 beta: float = 0.8,
                 max_load_per_robot: int = 20,
                 max_steps: int = 500,
                 seed: Optional[int] = 0):
        self.grid_size = grid_size
        self.n = n_robots
        self.init_battery = init_battery
        self.a = a
        self.b = b
        self.r = share_radius
        self.beta = beta
        self.max_load = max_load_per_robot
        self.max_steps = max_steps

        # storage nodes
        if storage_nodes is None:
            mid = (grid_size[0]//2, grid_size[1]//2)
            storage_nodes = {mid: 10}
        self.storage_nodes = dict(storage_nodes)

        # action and state sizes
        self.move_actions = 5
        self.w_bins = list(range(0, 11))
        self.per_robot_dims = [
            self.move_actions,
            2,
            self.n,
            grid_size[0] * grid_size[1],
            len(self.w_bins)
        ]

        # RNG and initial dest (may be overwritten in reset)
        self.rng = np.random.default_rng(seed)
        self._initial_dest = dest
        self.reset(seed)

    # -------------------------
    # Reset / Observations
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment. Destination is sampled uniformly anywhere on the grid,
        avoiding storage nodes and initial robot starting cells.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # initial robot positions on row 0, columns 0..n-1 (clipped)
        self.positions = {i: (0, min(i, self.grid_size[1] - 1)) for i in range(self.n)}
        self.loads = {i: 0 for i in range(self.n)}
        self.batteries = {i: self.init_battery for i in range(self.n)}
        self.inventory = dict(self.storage_nodes)
        self.time_step = 0
        self.done = False
        self.stuck_counters = {i: 0 for i in range(self.n)}

        # precompute nodes and radius-balls
        self._precompute()

        # sample destination uniformly (Option A). If an initial dest provided on init, still randomize on reset.
        self._sample_random_dest()

        # energy forecasts
        self.E_hat = {i: self._compute_energy_forecast(i) for i in range(self.n)}
        return self._get_obs()

    def _precompute(self):
        self.node_list = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])]
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.r_balls = {}
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.r_balls[(x, y)] = [
                    (i, j)
                    for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
                    if abs(x - i) + abs(y - j) <= self.r
                ]

    def _sample_random_dest(self):
        # valid cells exclude storage nodes and robot start positions
        forbidden = set(self.storage_nodes.keys()) | set(self.positions.values())
        candidates = [cell for cell in self.node_list if cell not in forbidden]
        if not candidates:
            # fallback: allow any cell
            candidates = list(self.node_list)
        self.dest = tuple(self.rng.choice(len(candidates)))
        # rng.choice returned index; convert:
        self.dest = candidates[self.dest]

    def _get_obs(self) -> Dict[str, Any]:
        per_robot = []
        for i in range(self.n):
            x, y = self.positions[i]
            load = self.loads[i]
            bat_ratio = float(self.batteries[i] / self.init_battery)
            Ehat = float(self.E_hat.get(i, 0.0))
            offload_hint = 1.0 if Ehat > self.beta * self.batteries[i] else 0.0
            per_robot.extend([
                x / self.grid_size[0],
                y / self.grid_size[1],
                load / self.max_load,
                bat_ratio,
                Ehat / (self.init_battery + 1e-8),
                offload_hint
            ])
        inv_vec = [self.inventory.get(n, 0) / 100.0 for n in self.node_list if n in self.storage_nodes]
        dest_vec = [self.dest[0] / self.grid_size[0], self.dest[1] / self.grid_size[1]]
        flat = np.array(per_robot + inv_vec + dest_vec, dtype=np.float32)
        return {"state": flat, "masks": self._compute_masks()}

    # -------------------------
    # Masks
    # -------------------------
    def _compute_masks(self) -> Dict[int, Dict[str, np.ndarray]]:
        masks = {i: {} for i in range(self.n)}
        for i in range(self.n):
            x, y = self.positions[i]
            move_mask = np.ones(self.move_actions, dtype=bool)
            if x == 0: move_mask[1] = False
            if x == self.grid_size[0] - 1: move_mask[2] = False
            if y == 0: move_mask[3] = False
            if y == self.grid_size[1] - 1: move_mask[4] = False
            if self.batteries[i] <= 0:
                move_mask[:] = False
                move_mask[0] = True
            masks[i]['move_mask'] = move_mask

        for i in range(self.n):
            if self.batteries[i] <= 0:
                masks[i]['partner_mask'] = np.zeros(self.n, dtype=bool)
                masks[i]['meeting_mask'] = np.zeros(len(self.node_list), dtype=bool)
                masks[i]['offload_mask'] = np.array([1, 0], dtype=bool)
                masks[i]['w_mask'] = np.zeros(len(self.w_bins), dtype=bool)
                continue
            partner_mask = np.zeros(self.n, dtype=bool)
            meeting_mask = np.zeros(len(self.node_list), dtype=bool)
            xi, yi = self.positions[i]
            for j in range(self.n):
                if i == j: continue
                if self.batteries[j] <= 0: continue
                xj, yj = self.positions[j]
                if abs(xi - xj) + abs(yi - yj) <= 2 * self.r:
                    partner_mask[j] = True
            for idx, node in enumerate(self.node_list):
                if node in self.r_balls[self.positions[i]]:
                    meeting_mask[idx] = True
            masks[i]['partner_mask'] = partner_mask
            masks[i]['meeting_mask'] = meeting_mask
            masks[i]['offload_mask'] = np.array([1, 1], dtype=bool)
            masks[i]['w_mask'] = np.array([1 if w <= self.loads[i] else 0 for w in self.w_bins], dtype=bool)
        return masks

    # -------------------------
    # Movement helpers
    # -------------------------
    def _apply_move(self, pos: Tuple[int,int], move: int, robot_idx: Optional[int] = None) -> Tuple[int,int]:
        if robot_idx is not None and self.batteries.get(robot_idx, 1) <= 0:
            return pos
        x, y = pos
        if move == 0: return (x, y)
        if move == 1 and x > 0: return (x - 1, y)
        if move == 2 and x < self.grid_size[0] - 1: return (x + 1, y)
        if move == 3 and y > 0: return (x, y - 1)
        if move == 4 and y < self.grid_size[1] - 1: return (x, y + 1)
        return (x, y)

    def _compute_energy_forecast(self, robot_idx: int, pos_override: Optional[Tuple[int,int]] = None) -> float:
        pos = pos_override if pos_override is not None else self.positions[robot_idx]
        path_len = abs(pos[0] - self.dest[0]) + abs(pos[1] - self.dest[1])
        W = self.loads[robot_idx]
        return float(path_len * (self.a * (W ** 2) + self.b))

    def _step_towards(self, pos: Tuple[int,int], target: Tuple[int,int]) -> Tuple[int,int]:
        x, y = pos; tx, ty = target
        if x < tx: return (x + 1, y)
        if x > tx: return (x - 1, y)
        if y < ty: return (x, y + 1)
        if y > ty: return (x, y - 1)
        return (x, y)

    # -------------------------
    # Step: core logic + detailed info
    # -------------------------
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        violations = {i: [] for i in range(self.n)}
        delivered = {i: False for i in range(self.n)}
        info['action_vector'] = action.copy() if isinstance(action, np.ndarray) else np.array(action)

        if self.done:
            obs = self._get_obs()
            info['state_vector'] = obs['state'].copy()
            info['violations'] = violations
            info['delivered'] = delivered
            info['energy'] = 0.0
            return obs, 0.0, True, info

        if action.shape[0] != self.n * len(self.per_robot_dims):
            raise ValueError("action length mismatch")

        parsed = {}
        for i in range(self.n):
            base = i * len(self.per_robot_dims)
            mv = int(action[base + 0])
            off = int(action[base + 1])
            partner = int(action[base + 2])
            meeting_idx = int(action[base + 3])
            w_idx = int(action[base + 4])
            parsed[i] = {"move": mv, "offload": off, "partner": partner, "meeting_idx": meeting_idx, "w_idx": w_idx}

        masks = self._compute_masks()

        # enforce masks; record violations
        for i in range(self.n):
            if not masks[i]['move_mask'][parsed[i]['move']]:
                violations[i].append("illegal_move_attempt")
                parsed[i]['move'] = 0
            if not masks[i]['offload_mask'][parsed[i]['offload']]:
                violations[i].append("illegal_offload_attempt")
                parsed[i]['offload'] = 0
            if parsed[i]['partner'] < 0 or parsed[i]['partner'] >= self.n:
                violations[i].append("invalid_partner_index")
                parsed[i]['partner'] = 0
            if parsed[i]['meeting_idx'] < 0 or parsed[i]['meeting_idx'] >= len(self.node_list):
                violations[i].append("invalid_meeting_index")
                parsed[i]['meeting_idx'] = self.node_to_idx[self.positions[i]]
            if parsed[i]['w_idx'] < 0 or parsed[i]['w_idx'] >= len(self.w_bins):
                violations[i].append("invalid_w_idx")
                parsed[i]['w_idx'] = 0

        # 1) apply moves
        prev_positions = dict(self.positions)
        for i in range(self.n):
            self.positions[i] = self._apply_move(self.positions[i], parsed[i]['move'], robot_idx=i)

        # collisions -> revert and mark violation
        pos_counts = {}
        for i in range(self.n):
            pos_counts.setdefault(self.positions[i], []).append(i)
        collisions = []
        for pos, who in pos_counts.items():
            if len(who) > 1:
                collisions.extend(who)
                for idx in who:
                    self.positions[idx] = prev_positions[idx]
                    violations[idx].append("collision")

        # 2) pickup / delivery
        for i in range(self.n):
            pos = self.positions[i]
            if pos in self.inventory and self.inventory[pos] > 0 and self.loads[i] < self.max_load and self.batteries[i] > 0:
                pickup = min(1, self.inventory[pos], self.max_load - self.loads[i])
                self.inventory[pos] -= pickup
                self.loads[i] += pickup
            if pos == self.dest and self.loads[i] > 0:
                self.loads[i] = 0
                delivered[i] = True

    # -------------------------
    # (continued) sharing, energy, penalties, return
    # -------------------------
        # 3) collect sharing proposals
        proposals = []
        for i in range(self.n):
            if parsed[i]['offload'] == 1 and self.batteries[i] > 0:
                j = parsed[i]['partner']
                if j == i:
                    violations[i].append("offload_to_self")
                    continue
                if self.batteries.get(j, -1) <= 0:
                    violations[i].append("partner_no_battery")
                    continue
                if abs(self.positions[i][0] - self.positions[j][0]) + abs(self.positions[i][1] - self.positions[j][1]) > 2 * self.r:
                    violations[i].append("partner_out_of_range")
                    continue
                meeting_node = self.idx_to_node.get(parsed[i]['meeting_idx'], self.positions[i])
                if meeting_node not in self.r_balls[self.positions[i]] or meeting_node not in self.r_balls[self.positions[j]]:
                    violations[i].append("meeting_node_infeasible")
                    continue
                Wi = self.loads[i]; Wj = self.loads[j]
                w = (Wi - Wj) // 2
                if w <= 0:
                    violations[i].append("no_need_to_share")
                    continue
                w = min(w, self.max_load - self.loads[j])
                if w <= 0:
                    violations[i].append("receiver_capacity_full")
                    continue
                proposals.append((i, j, meeting_node, w))

        # 4) resolve proposals (move toward meeting node, transfer if co-located)
        sharing_events = []
        for (i, j, meeting_node, w) in proposals:
            if self.positions[i] == meeting_node and self.positions[j] == meeting_node:
                self.loads[i] -= w
                self.loads[j] += w
                sharing_events.append((i, j, meeting_node, w))
            else:
                self.positions[i] = self._step_towards(self.positions[i], meeting_node)
                self.positions[j] = self._step_towards(self.positions[j], meeting_node)
                if self.positions[i] == self.positions[j] == meeting_node:
                    self.loads[i] -= w
                    self.loads[j] += w
                    sharing_events.append((i, j, meeting_node, w))

        # 5) energy update and penalties
        total_energy = 0.0
        penalty = 0.0
        for i in range(self.n):
            d = abs(prev_positions[i][0] - self.positions[i][0]) + abs(prev_positions[i][1] - self.positions[i][1])
            step_d = d if d > 0 else 1
            e = self.a * (self.loads[i] ** 2) * step_d + self.b
            self.batteries[i] -= e
            total_energy += e
            # forecast infeasibility
            self.E_hat[i] = self._compute_energy_forecast(i)
            if self.E_hat[i] > self.batteries[i]:
                violations[i].append("forecast_infeasible")
                penalty -= 10
            if parsed[i]['move'] != 0 and prev_positions[i] == self.positions[i] and "collision" not in violations[i]:
                violations[i].append("illegal_move_no_effect")
                penalty -= 1
            if self.positions[i] == prev_positions[i]:
                self.stuck_counters[i] += 1
            else:
                self.stuck_counters[i] = 0
            if self.stuck_counters[i] > 5:
                violations[i].append("stuck")
                penalty -= 3

        for i in range(self.n):
            if self.batteries[i] < 0:
                violations[i].append("battery_depleted")
                penalty -= 50

        for idx in collisions:
            penalty -= 5

        reward = - total_energy + penalty

        all_items_delivered = all(v == 0 for v in self.inventory.values()) and all(self.loads[i] == 0 for i in range(self.n))
        if all_items_delivered:
            reward += 100
            self.done = True

        self.time_step += 1
        if self.time_step >= self.max_steps:
            self.done = True

        obs = self._get_obs()
        info['state_vector'] = obs['state'].copy()
        info['violations'] = violations
        info['delivered'] = delivered
        info['sharing_events'] = sharing_events
        info['collisions'] = collisions
        info['energy'] = total_energy
        info['reward'] = float(reward)
        return obs, float(reward), bool(self.done), info
