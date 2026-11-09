"""Tracking + pose-graph SLAM front/back end with per-pose uncertainty.

Prototype backend: Open3D hybrid photometric/geometric RGB-D odometry,
frame-to-keyframe tracking, radius-search loop closures and pose-graph
optimisation.  The interface (poses + covariances per frame) is the same one a
VIGS-SLAM run feeds in through `load_trajectory()`, so the downstream
GASF / mapping / twin modules are backend-agnostic.
"""
import time

import cv2
import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


def make_rgbd(frame, depth_max):
    color = o3d.geometry.Image(np.ascontiguousarray(frame["rgb"]))
    depth = o3d.geometry.Image(np.ascontiguousarray(frame["depth"].astype(np.float32)))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=depth_max, convert_rgb_to_intensity=True)


def o3d_intrinsic(K, w, h):
    return o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def _rot_angle(T):
    return np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))


def pose_sigma(info):
    """Scalar 1-sigma pose uncertainty (metres-equivalent) from a 6x6 information matrix."""
    try:
        cov = np.linalg.inv(info + 1e-9 * np.eye(6))
        return float(np.sqrt(max(np.trace(cov[3:, 3:]), 0)) + 0.5 * np.sqrt(max(np.trace(cov[:3, :3]), 0)))
    except np.linalg.LinAlgError:
        return 1.0


class PnPInit:
    """ORB + PnP-RANSAC relative pose (robust initialiser for dense odometry)."""

    def __init__(self, K, n_features=1500, depth_max=8.0):
        self.K = K.astype(np.float64); self.depth_max = depth_max
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, frame):
        g = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(g, None)
        return kp, des, frame["depth"]

    def relative(self, fa, fb):
        """Return T_b_a (maps points of frame a into frame b) and inlier count, or (None, 0)."""
        kpa, da, depth_a = fa
        kpb, db, _ = fb
        if da is None or db is None or len(kpa) < 20 or len(kpb) < 20:
            return None, 0
        good = []
        for m in self.bf.knnMatch(da, db, k=2):
            if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
                good.append(m[0])
        if len(good) < 15:
            return None, 0
        pa = np.array([kpa[m.queryIdx].pt for m in good]); pb = np.array([kpb[m.trainIdx].pt for m in good])
        z = depth_a[pa[:, 1].astype(int), pa[:, 0].astype(int)]
        ok = (z > 0.2) & (z < self.depth_max)
        if ok.sum() < 12:
            return None, 0
        X = np.stack([(pa[ok, 0] - self.K[0, 2]) / self.K[0, 0] * z[ok],
                      (pa[ok, 1] - self.K[1, 2]) / self.K[1, 1] * z[ok], z[ok]], 1)
        res = cv2.solvePnPRansac(X, pb[ok], self.K, None, iterationsCount=200, reprojectionError=2.0,
                                 confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not res[0] or res[3] is None or len(res[3]) < 12:
            return None, 0
        R, _ = cv2.Rodrigues(res[1])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = res[2].ravel()
        return T, len(res[3])


def _delta(A, B):
    E = np.linalg.inv(A) @ B
    return np.linalg.norm(E[:3, 3]), _rot_angle(E)


def depth_consistency(depth_a, depth_b, K, T_b_a, stride=6, tol=0.04, depth_max=8.0):
    """Fraction of frame-a depth points that land on frame-b's depth surface after T_b_a."""
    h, w = depth_a.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_a[::stride, ::stride]
    m = (z > 0.2) & (z < depth_max)
    if m.sum() < 50:
        return 0.0
    P = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
    Q = P @ T_b_a[:3, :3].T + T_b_a[:3, 3]
    ok = Q[:, 2] > 0.2
    uu = np.round(K[0, 0] * Q[ok, 0] / Q[ok, 2] + K[0, 2]).astype(int)
    vv = np.round(K[1, 1] * Q[ok, 1] / Q[ok, 2] + K[1, 2]).astype(int)
    inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    if inb.sum() < 50:
        return 0.0
    zb = depth_b[vv[inb], uu[inb]]
    good = (zb > 0) & (np.abs(zb - Q[ok][inb, 2]) < tol * np.maximum(1.0, Q[ok][inb, 2] / 3))
    return float(good.sum() / m.sum())


class RGBDSLAM:
    """tracker: 'hybrid' (dense photometric+geometric), 'icp' (point-to-plane ICP on the depth cloud)
    or 'hybrid+icp' (dense result refined by ICP). use_pnp / use_loops switch components off for ablations."""

    def __init__(self, K, width, height, depth_max=6.0, kf_trans=0.20, kf_rot=12.0,
                 loop_radius=2.0, loop_min_gap=15, depth_diff_max=0.07, tracker="hybrid", use_pnp=True,
                 use_loops=True, icp_voxel=None):
        self.tracker, self.use_pnp, self.use_loops = tracker, use_pnp, use_loops
        self.icp_voxel = icp_voxel or max(0.02, depth_max / 200.0)
        self.intr = o3d_intrinsic(K, width, height)
        self.K = np.asarray(K, float)
        self.depth_max = depth_max
        self.opt = o3d.pipelines.odometry.OdometryOption(
            iteration_number_per_pyramid_level=o3d.utility.IntVector([20, 10, 5]),
            depth_diff_max=depth_diff_max, depth_min=0.2, depth_max=depth_max)
        self.jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self.kf_trans, self.kf_rot = kf_trans, kf_rot
        self.loop_radius, self.loop_min_gap = loop_radius, loop_min_gap
        self.check_max = max(8.0, depth_max)        # range used by PnP / consistency checks
        self.pnp = PnPInit(K, depth_max=self.check_max)
        self.keyframes = []          # dict(idx, rgbd, T, feat)
        self.rel = []                # per frame: (kf index, T_kf_frame)
        self.poses, self.sigmas = [], []
        self.step_sigmas = []        # per-frame tracking uncertainty (used for gating / fusion weights)
        self.graph = reg.PoseGraph()
        self.n_loops, self.n_track_fail = 0, 0
        self.timing = []

    def _cloud(self, rgbd):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intr).voxel_down_sample(self.icp_voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * self.icp_voxel, max_nn=30))
        return pc

    def _icp(self, src, tgt, init):
        a, b = self._cloud(src), self._cloud(tgt)
        T = init
        for d in (8 * self.icp_voxel, 3 * self.icp_voxel, 1.5 * self.icp_voxel):
            r = reg.registration_icp(a, b, d, T, reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=30))
            T = r.transformation
        info = reg.get_information_matrix_from_point_clouds(a, b, 1.5 * self.icp_voxel, T)
        return r.fitness > 0.3, T, info

    def _odom(self, src, tgt, init):
        if self.tracker == "icp":
            return self._icp(src, tgt, init)
        ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(src, tgt, self.intr, init, self.jac, self.opt)
        if self.tracker == "hybrid+icp":
            ok2, T2, info2 = self._icp(src, tgt, T if ok else init)
            if ok2:
                return True, T2, info2
        return ok, T, info

    def track(self, frame):
        t0 = time.perf_counter()
        rgbd = make_rgbd(frame, self.depth_max)
        feat = self.pnp.features(frame)
        i = len(self.poses)
        if i == 0:
            T = np.eye(4)
            self.keyframes.append(dict(idx=0, rgbd=rgbd, T=T, feat=feat, depth=frame["depth"]))
            self.graph.nodes.append(reg.PoseGraphNode(T))
            self.rel.append((0, np.eye(4))); self.poses.append(T); self.sigmas.append(0.0); self.step_sigmas.append(0.0)
            self.timing.append(time.perf_counter() - t0)
            return T
        kf = self.keyframes[-1]
        # constant-velocity prediction expressed relative to the keyframe
        T_pred = self.poses[-1] if i < 2 else self.poses[-1] @ np.linalg.inv(self.poses[-2]) @ self.poses[-1]
        init_cv = np.linalg.inv(T_pred) @ kf["T"]                   # constant-velocity prior (kf -> current)
        best, best_score, info = None, -1.0, None

        def consider(init):
            nonlocal best, best_score, info
            ok, T_try, inf = self._odom(kf["rgbd"], rgbd, init)
            for T_c, inf_c in (((T_try, inf), (init, None)) if ok else ((init, None),)):
                sc = depth_consistency(kf["depth"], frame["depth"], self.K, T_c, depth_max=self.check_max)
                if sc > best_score:
                    best, best_score, info = T_c, sc, inf_c

        consider(init_cv)
        if self.use_pnp and (best_score < 0.75 or info is None):          # weak: try the feature-based (ORB + PnP) initialisation
            T_pnp, n_in = self.pnp.relative(kf["feat"], feat)
            if T_pnp is not None and n_in >= 15:
                consider(T_pnp)
        T_cur_kf = best
        if info is None or best_score < 0.35:
            self.n_track_fail += 1
            info = np.eye(6) * max(best_score, 0.05) * 1e3
        T_kf_cur = np.linalg.inv(T_cur_kf)
        T = kf["T"] @ T_kf_cur
        step = min(pose_sigma(info), 0.5)
        sig = self.sigmas[kf["idx"]] + step                  # accumulated (dead-reckoning) uncertainty
        self.rel.append((len(self.keyframes) - 1, T_kf_cur)); self.poses.append(T); self.sigmas.append(sig)
        self.step_sigmas.append(step)
        if np.linalg.norm(T_kf_cur[:3, 3]) > self.kf_trans or _rot_angle(T_kf_cur) > self.kf_rot:
            self._add_keyframe(i, rgbd, T, T_kf_cur, info, feat, frame["depth"])
        self.timing.append(time.perf_counter() - t0)
        return T

    def _add_keyframe(self, i, rgbd, T, T_prev_cur, info, feat, depth):
        k = len(self.keyframes)
        self.keyframes.append(dict(idx=i, rgbd=rgbd, T=T, feat=feat, depth=depth))
        self.graph.nodes.append(reg.PoseGraphNode(T))
        # Open3D edge convention: transformation maps source-frame points into the target frame
        self.graph.edges.append(reg.PoseGraphEdge(k - 1, k, np.linalg.inv(T_prev_cur), info, uncertain=False))
        self.rel[i] = (k, np.eye(4))
        if not self.use_loops:
            return
        # loop closure against older nearby keyframes
        c = T[:3, 3]
        for j, kj in enumerate(self.keyframes[:-self.loop_min_gap]):
            if np.linalg.norm(kj["T"][:3, 3] - c) > self.loop_radius:
                continue
            init = np.linalg.inv(T) @ kj["T"]
            T_pnp, n_in = self.pnp.relative(kj["feat"], feat)
            if T_pnp is None or n_in < 40:
                continue
            ok, T_cur_j, inf = self._odom(kj["rgbd"], rgbd, T_pnp)
            if not ok or depth_consistency(kj["depth"], depth, self.K, T_cur_j, depth_max=self.check_max) < 0.5:
                continue
            if np.linalg.norm((init @ np.linalg.inv(T_cur_j))[:3, 3]) > self.loop_radius + 0.5:   # implausible
                continue
            self.graph.edges.append(reg.PoseGraphEdge(j, k, T_cur_j, inf, uncertain=True))
            self.n_loops += 1

    def optimize(self):
        self.poses_before_opt = [T.copy() for T in self.poses]
        if self.n_loops == 0:
            return self.poses
        option = reg.GlobalOptimizationOption(max_correspondence_distance=0.05,
                                              edge_prune_threshold=0.25, reference_node=0)
        reg.global_optimization(self.graph, reg.GlobalOptimizationLevenbergMarquardt(),
                                reg.GlobalOptimizationConvergenceCriteria(), option)
        for k, node in enumerate(self.graph.nodes):
            self.keyframes[k]["T"] = np.asarray(node.pose)
        self.poses = [self.keyframes[k]["T"] @ Trel for (k, Trel) in self.rel]
        return self.poses


def run_slam(seq, depth_max=6.0, verbose=True, **kw):
    f0 = seq.frame(0)
    h, w = f0["depth"].shape
    slam = RGBDSLAM(seq.K, w, h, depth_max=depth_max, **kw)
    for i in range(len(seq)):
        slam.track(seq.frame(i))
        if verbose and i % 50 == 0:
            print(f"  tracked {i}/{len(seq)}  keyframes={len(slam.keyframes)}  loops={slam.n_loops}")
    t0 = time.perf_counter()
    last = len(slam.poses) - 1
    if slam.rel[last][0] != len(slam.keyframes) - 1 or not np.allclose(slam.rel[last][1], np.eye(4)):
        f = seq.frame(last)                                  # close the trajectory with a final keyframe
        k, T_k_cur = slam.rel[last]
        slam._add_keyframe(last, make_rgbd(f, slam.depth_max), slam.poses[last], T_k_cur, np.eye(6) * 1e3,
                           slam.pnp.features(f), f["depth"])
    poses = slam.optimize()
    slam.t_opt = time.perf_counter() - t0
    return slam, poses


def load_trajectory(path, timestamps, max_dt=0.03):
    """Load a TUM-format trajectory (e.g. VIGS-SLAM traj_*.txt) and associate it to frames."""
    from .data import quat_to_R
    rows = np.loadtxt(path)
    out = []
    for t in timestamps:
        j = int(np.argmin(np.abs(rows[:, 0] - t)))
        tx, ty, tz, qx, qy, qz, qw = rows[j, 1:8]
        T = np.eye(4); T[:3, :3] = quat_to_R(qw, qx, qy, qz); T[:3, 3] = (tx, ty, tz)
        out.append(T if abs(rows[j, 0] - t) < max_dt else None)
    return out
