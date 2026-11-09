"""Uncertainty-guided next-best-view exploration with LiDAR obstacle checks and
local replanning, plus frontier / random baselines (run in the synthetic simulator)."""
import heapq

import numpy as np
from scipy import ndimage

from .mapping import OccupancyMap


def dijkstra(free, start):
    """8-connected grid distances (cells) from start over free cells."""
    H, W = free.shape
    dist = np.full((H, W), np.inf); prev = -np.ones((H, W, 2), int)
    if not free[start]:
        return dist, prev
    dist[start] = 0; pq = [(0.0, start)]
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, c in nb:
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and free[a, b] and d + c < dist[a, b]:
                dist[a, b] = d + c; prev[a, b] = (i, j); heapq.heappush(pq, (d + c, (a, b)))
    return dist, prev


def backtrack(prev, goal):
    path = [goal]
    while True:
        p = tuple(prev[path[-1]])
        if p[0] < 0:
            break
        path.append(p)
    return path[::-1]


def camera_pose(x, y, yaw, height=1.4, pitch_deg=-12):
    from .synthetic import look_at
    d = np.array([np.cos(yaw), np.sin(yaw), np.tan(np.radians(pitch_deg))])
    eye = np.array([x, y, height])
    return look_at(eye, eye + d)


NBV_WEIGHTS = {"nbv": (1.0, 2.0, 4.0),            # ours: unknown space + uncertain surfaces + ambiguous labels
               "nbv_no_semantic": (1.0, 2.0, 0.0),
               "nbv_coverage_only": (1.0, 0.0, 0.0)}   # classic volumetric information gain


class NBVPlanner:
    """Next-best-view planner on a live OccupancyMap (+ optional GASF for label ambiguity).

    utility(view) = (w_u * unknown voxels seen + w_s * uncertain surface voxels hit + w_a * ambiguous-asset
    voxels hit) * exp(-lambda * path length), evaluated for sampled reachable poses x 8 headings."""

    def __init__(self, occ, gasf=None, robot_radius=0.35, floor_z=0.0, robot_height=1.8, cam_height=1.4,
                 weights=(1.0, 2.0, 4.0), lam=0.15, n_candidates=60, seed=0):
        self.occ, self.gasf = occ, gasf
        self.r_cells = int(np.ceil(robot_radius / occ.res))
        self.floor_z, self.robot_height, self.cam_height = floor_z, robot_height, cam_height
        self.weights, self.lam, self.n_candidates = weights, lam, n_candidates
        self.rng = np.random.default_rng(seed)

    def nav_grid(self):
        trav = self.occ.traversability(floor_z=self.floor_z, robot_height=self.robot_height)
        blocked = ndimage.binary_dilation(trav == 0, iterations=self.r_cells)
        return (trav == 1) & ~blocked

    def ambiguous_voxels(self):
        if self.gasf is None:
            return np.empty(0, np.int64)
        P = [i.pts for i in self.gasf.instances.values()
             if (not self.gasf.is_confirmed(i)) or i.label_entropy() > 0.4]
        if not P:
            return np.empty(0, np.int64)
        ijk, ok = self.occ.index(np.vstack(P))
        return np.unique(self.occ.flat(ijk[ok]))

    def _cam(self, x, y, yaw):
        return camera_pose(x, y, yaw, height=self.floor_z + self.cam_height)

    def view_gain(self, x, y, yaw, state, sigma, amb, hfov=62, vfov=48, nh=20, nv=10, rng_max=6.0):
        T = self._cam(x, y, yaw)
        a = np.radians(np.linspace(-hfov / 2, hfov / 2, nh)); b = np.radians(np.linspace(-vfov / 2, vfov / 2, nv))
        A, B = np.meshgrid(a, b)
        dirs_c = np.stack([np.tan(A.ravel()), np.tan(B.ravel()), np.ones(A.size)], 1)
        dirs_c /= np.linalg.norm(dirs_c, axis=1, keepdims=True)
        dirs = dirs_c @ T[:3, :3].T
        steps = np.arange(0.3, rng_max, self.occ.res)
        S = T[:3, 3] + dirs[:, None, :] * steps[None, :, None]
        ijk, ok = self.occ.index(S.reshape(-1, 3))
        flat = np.where(ok, self.occ.flat(np.where(ok[:, None], ijk, 0)), -1).reshape(len(dirs), len(steps))
        st = np.where(flat >= 0, state[np.maximum(flat, 0)], 1)
        occ_hit = st == 1
        first = np.where(occ_hit.any(1), occ_hit.argmax(1), len(steps))
        idx = np.arange(len(steps))[None, :]
        before = idx < first[:, None]
        unknown = np.unique(flat[before & (st == -1)])
        hit_flat = flat[np.arange(len(dirs)), np.minimum(first, len(steps) - 1)][first < len(steps)]
        unc = np.sum(sigma[hit_flat] > 0.02) if len(hit_flat) else 0
        sem = np.isin(hit_flat, amb).sum() if len(amb) else 0
        return len(unknown), unc, sem

    def best_view(self, reach, dist):
        state = self.occ.state(); sigma = self.occ.surface_sigma(); amb = self.ambiguous_voxels()
        cand = reach[self.rng.choice(len(reach), min(self.n_candidates, len(reach)), replace=False)]
        wu, ws, wa = self.weights
        best, bu, parts = None, -1, None
        for (i, j) in cand:
            x, y = self.occ.center(np.array([[i, j, 0]]))[0][:2]
            for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                gu, gs, gm = self.view_gain(x, y, yaw, state, sigma, amb)
                u = (wu * gu + ws * gs + wa * gm) * np.exp(-self.lam * dist[i, j] * self.occ.res)
                if u > bu:
                    best, bu, parts = ((i, j), yaw), u, (gu, gs, gm)
        return best, bu, parts

    def plan(self, x, y, known_free_xy=None, blind_radius=1.5):
        """Plan from robot position (x, y). Returns dict(goal=(x, y), yaw, utility, gains, path=[(x, y), ...]) or None.
        known_free_xy: positions the robot has already occupied (its trajectory) - free by construction, which also
        clears obstacle ghosts left by pose drift. blind_radius: the near-field blind zone around the robot is treated
        as free unless an obstacle was seen there."""
        trav = self.occ.traversability(floor_z=self.floor_z, robot_height=self.robot_height).copy()
        H, W = trav.shape
        ri, rj, _ = self.occ.index(np.array([[x, y, self.floor_z + 0.5]]))[0][0]
        ri, rj = int(np.clip(ri, 0, H - 1)), int(np.clip(rj, 0, W - 1))
        ii, jj = np.mgrid[0:H, 0:W]
        rr = max(self.r_cells, 1)
        swept = (ii - ri) ** 2 + (jj - rj) ** 2 <= rr * rr
        if known_free_xy is not None and len(known_free_xy):
            ij, ok = self.occ.index(np.column_stack([np.asarray(known_free_xy)[:, :2],
                                                     np.full(len(known_free_xy), self.floor_z + 0.5)]))
            sw = np.zeros_like(swept)
            sw[np.clip(ij[ok, 0], 0, H - 1), np.clip(ij[ok, 1], 0, W - 1)] = True
            swept |= ndimage.binary_dilation(sw, iterations=rr)
        trav[swept] = 1                                          # the robot has physically been here
        rc = int(blind_radius / self.occ.res)
        blind = (ii - ri) ** 2 + (jj - rj) ** 2 <= rc * rc
        trav[blind & (trav == -1)] = 1
        blocked = ndimage.binary_dilation(trav == 0, iterations=self.r_cells)
        nav = (trav == 1) & ~blocked
        nav[swept] = True
        dist, prev = dijkstra(nav, (int(ri), int(rj)))
        reach = np.argwhere(np.isfinite(dist) & (dist > 3))
        if len(reach) == 0:
            return None
        (g, yaw), u, parts = self.best_view(reach, dist)
        path = [tuple(self.occ.center(np.array([[c[0], c[1], 0]]))[0][:2]) for c in backtrack(prev, g)]
        return dict(goal=path[-1], yaw=float(yaw), utility=float(u), gains=dict(unknown=int(parts[0]),
                    uncertain=int(parts[1]), ambiguous=int(parts[2])), path=path)


class Explorer(NBVPlanner):
    def __init__(self, sim, strategy="nbv", res=0.1, bounds=((-0.5, -0.5, -0.2), (24.5, 16.5, 6.0)),
                 robot_radius=0.35, seed=0, gasf=None, detector=None, classes=None):
        self.sim, self.strategy = sim, strategy
        occ = OccupancyMap(bounds[0], bounds[1], res, max_range=7.0)
        super().__init__(occ, gasf, robot_radius=robot_radius, weights=NBV_WEIGHTS.get(strategy, (1.0, 2.0, 4.0)),
                         seed=seed)
        self.gasf, self.det, self.classes = gasf, detector, classes
        self.asset_pixels = {}
        self.log = []
        self.path_len = 0.0
        self.replans = 0
        self.collisions = 0
        self.seg_counter = 0
        self.visited_goals = []
        # GT surface for coverage
        import open3d as o3d
        pts = []
        for o in sim.objs:
            if o["cls"] in ("floor",):
                continue
            pc = o["mesh"].sample_points_uniformly(int(max(200, o["mesh"].get_surface_area() * 40)))
            pts.append(np.asarray(pc.points))
        P = np.vstack(pts)
        P = P[P[:, 2] < 4.5]
        ijk, ok = self.occ.index(P)
        self.gt_surface = np.unique(self.occ.flat(ijk[ok]))
        # GT 2D obstacle map for collision accounting
        occ_gt = np.zeros(self.occ.shape[:2], bool)
        Q = P[(P[:, 2] > 0.1) & (P[:, 2] < 1.8)]
        ij, ok2 = self.occ.index(Q)
        occ_gt[ij[ok2, 0], ij[ok2, 1]] = True
        self.gt_obst = occ_gt

    # ---------------------------------------------------------------- sensing
    def lidar(self, x, y, z=0.35, n=360, rmax=8.0):
        import open3d as o3d
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1)
        rays = np.concatenate([np.tile([x, y, z], (n, 1)), d], 1).astype(np.float32)
        t = self.sim.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(t) & (t < rmax)
        return np.array([x, y, z]) + d[hit] * t[hit, None]

    def sense(self, x, y, yaw, frame_idx):
        T = camera_pose(x, y, yaw)
        f = self.sim.render(T)
        self.occ.integrate(f["depth"], self.sim.K, T, stride=6)
        for z in (0.08, 0.3, 0.6, 1.0):                             # multi-ring LiDAR
            L = self.lidar(x, y, z=z)
            self.occ.integrate_points(L, np.array([x, y, z]))
        ids, cnt = np.unique(f["instance"], return_counts=True)
        for i, c in zip(ids, cnt):
            if i:
                self.asset_pixels[i] = self.asset_pixels.get(i, 0) + c
        if self.gasf is not None:
            from .gasf import make_segments
            segs = make_segments(f, self.det(f), self.sim.K, T, self.classes, frame_idx,
                                 gt_instance=f["instance"], start_sid=self.seg_counter)
            self.seg_counter += len(segs)
            self.gasf.update(segs, f, self.sim.K, T)

    # ---------------------------------------------------------------- planning
    def choose_goal(self, ri, rj, nav):
        dist, prev = dijkstra(nav, (ri, rj))
        reach = np.argwhere(np.isfinite(dist) & (dist > 3))
        if len(reach) == 0:
            return None, None, dist, prev
        state = self.occ.state(); sigma = self.occ.surface_sigma()
        if self.strategy == "random":
            g = tuple(reach[self.rng.integers(len(reach))])
            return g, self.rng.uniform(0, 2 * np.pi), dist, prev
        st2 = state.reshape(self.occ.shape)
        unknown_col = (st2[:, :, 1:20] == -1).sum(axis=2) > 5
        if self.strategy == "frontier":
            front = nav & ndimage.binary_dilation(unknown_col, iterations=2) & np.isfinite(dist)
            cand = np.argwhere(front & (dist > 3))
            if len(cand) and self.visited_goals:                    # classic frontier blacklisting
                V = np.array(self.visited_goals)
                dmin = np.min(np.linalg.norm(cand[:, None, :] - V[None, :, :], axis=2), axis=1)
                cand = cand[dmin * self.occ.res > 1.5]
            if len(cand) == 0:
                g = tuple(reach[self.rng.integers(len(reach))]); return g, self.rng.uniform(0, 2 * np.pi), dist, prev
            g = tuple(cand[np.argmin(dist[tuple(cand.T)])])
            uc = np.argwhere(unknown_col)
            near = uc[np.argmin(np.linalg.norm(uc - np.array(g), axis=1))] if len(uc) else np.array(g) + 1
            return g, float(np.arctan2(near[1] - g[1], near[0] - g[0])), dist, prev
        # NBV variants: sample candidates, evaluate weighted information gain of 8 headings
        best, _, _ = self.best_view(reach, dist)
        return best[0], best[1], dist, prev

    # ---------------------------------------------------------------- main loop
    def run(self, start=(3.5, 8.0, 0.0), budget_m=80.0, max_steps=400, step_m=0.5):
        x, y, yaw = start
        k = 0
        for a in np.linspace(0, 2 * np.pi, 4, endpoint=False):       # initial look-around
            self.sense(x, y, yaw + a, k); k += 1
        self._log(k)
        while self.path_len < budget_m and k < max_steps:
            nav = self.nav_grid()
            ri, rj, _ = self.occ.index(np.array([[x, y, 0.5]]))[0][0]
            nav[max(ri - 1, 0):ri + 2, max(rj - 1, 0):rj + 2] = True
            goal, gyaw, dist, prev = self.choose_goal(ri, rj, nav)
            if goal is None:
                break
            self.visited_goals.append(goal)
            path = backtrack(prev, goal)
            pos = np.array([x, y]); acc = 0.0
            for n, cell in enumerate(path[1:], 1):
                c = self.occ.center(np.array([[cell[0], cell[1], 0]]))[0][:2]
                seg = np.linalg.norm(c - pos); acc += seg; self.path_len += seg
                heading = np.arctan2(c[1] - pos[1], c[0] - pos[0])
                pos = c
                if self.gt_obst[cell[0], cell[1]]:
                    self.collisions += 1
                if acc >= step_m or n == len(path) - 1:
                    acc = 0.0
                    look = gyaw if n == len(path) - 1 else heading
                    self.sense(pos[0], pos[1], look, k); k += 1
                    self._log(k)
                    # local replanning: LiDAR/camera found an obstacle on the remaining path
                    nav2 = self.nav_grid()
                    rest = path[n + 1:]
                    if any(not nav2[c2] for c2 in rest):
                        self.replans += 1
                        break
                if self.path_len >= budget_m:
                    break
            x, y = pos
            yaw = gyaw
        return self.log

    def _log(self, k):
        seen = sum(1 for v in self.asset_pixels.values() if v > 1500)
        self.log.append(dict(step=k, path_m=self.path_len, coverage=self.occ.coverage(self.gt_surface),
                             assets_seen=seen, replans=self.replans, collisions=self.collisions))
