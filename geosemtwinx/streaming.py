"""Incremental (online) GeoSemTwin-X: feed RGB-D frames one at a time, get a live digital twin.

Used by the ROS 2 node and by `scripts/run_all.py demo`. Frame dict: rgb (HxWx3 uint8), depth (HxW float32, metres).
World frame = T_init @ SLAM frame. Give T_init so that z points up and the floor is at floor_z (e.g. the first
ground-truth pose, or the camera mounting pose from the robot config) to enable traversability and NBV planning.
"""
import time

import numpy as np

from .gasf import GASF, make_segments
from .mapping import FusedMap, OccupancyMap
from .nbv import NBVPlanner
from .slam import RGBDSLAM
from .twin import DigitalTwin, TwinStore


def mount_pose(height=1.0, pitch_deg=0.0):
    """Camera-to-world pose for a forward-looking camera mounted at `height` (world: x forward, y left, z up)."""
    R_opt_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    p = np.radians(pitch_deg)                                    # positive = tilt down
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    T = np.eye(4); T[:3, :3] = Ry @ R_opt_to_body; T[:3, 3] = (0, 0, height)
    return T


class LiveTwin:
    def __init__(self, K, width, height, classes, detector=None, T_init=None, detect_every=5, depth_max=6.0,
                 map_voxel=0.03, map_sigma_max=0.05, occ_res=0.1, map_extent=20.0, floor_z=0.0, floor_classes=(),
                 min_score=0.30, tracker="hybrid", nbv=True, robot_radius=0.35, cam_height=None):
        self.K = np.asarray(K, float); self.classes = list(classes); self.detector = detector
        self.T_init = np.eye(4) if T_init is None else np.asarray(T_init, float)
        self.detect_every, self.depth_max, self.floor_z = detect_every, depth_max, floor_z
        self.seg_kw = dict(floor_classes=tuple(floor_classes), floor_z=floor_z, min_score=min_score)
        self.slam = RGBDSLAM(K, width, height, depth_max=depth_max, tracker=tracker)
        self.fmap = FusedMap(K, width, height, voxel=map_voxel, depth_max=depth_max, sigma_max=map_sigma_max)
        c = self.T_init[:3, 3]
        lo = np.array([c[0] - map_extent, c[1] - map_extent, floor_z - 0.5])
        hi = np.array([c[0] + map_extent, c[1] + map_extent, floor_z + 6.0])
        self.occ = OccupancyMap(lo, hi, occ_res, max_range=depth_max)
        self.gasf = GASF(self.classes)
        self.planner = NBVPlanner(self.occ, self.gasf, robot_radius=robot_radius, floor_z=floor_z,
                                  cam_height=cam_height if cam_height is not None else max(c[2] - floor_z, 0.3)) if nbv else None
        self.i, self.n_segments, self.n_dets = 0, 0, 0
        self.timing = dict(track=[], map=[], occ=[], detect=[], lift=[], fuse=[])
        self.last_pose, self.last_sigma = self.T_init.copy(), 0.0
        self.traj = []

    # ------------------------------------------------------------------ main step
    def process(self, frame):
        t = time.perf_counter()
        T_slam = self.slam.track(frame)
        T = self.T_init @ T_slam
        sig = self.slam.step_sigmas[-1]
        self.timing["track"].append(time.perf_counter() - t)
        t = time.perf_counter(); self.fmap.integrate(frame, T, sig); self.timing["map"].append(time.perf_counter() - t)
        keyframe = self.i % self.detect_every == 0
        dets = []
        if keyframe:
            t = time.perf_counter(); self.occ.integrate(frame["depth"], self.K, T, stride=8, pose_sigma=sig)
            self.timing["occ"].append(time.perf_counter() - t)
            if self.detector is not None:
                t = time.perf_counter(); dets = self.detector(frame["rgb"]); self.timing["detect"].append(time.perf_counter() - t)
                t = time.perf_counter()
                segs = make_segments(frame, dets, self.K, T, self.classes, self.i, pose_sigma=sig, max_depth=self.depth_max,
                                     gt_instance=frame.get("instance"), start_sid=self.n_segments, **self.seg_kw)
                self.timing["lift"].append(time.perf_counter() - t)
                t = time.perf_counter(); self.gasf.update(segs, frame, self.K, T); self.timing["fuse"].append(time.perf_counter() - t)
                self.n_segments += len(segs); self.n_dets += len(dets)
        self.last_pose, self.last_sigma = T, sig
        self.traj.append(T[:3, 3].copy())
        self.i += 1
        return dict(pose=T, sigma=sig, keyframe=keyframe, detections=dets)

    # ------------------------------------------------------------------ outputs
    def assets(self):
        out = []
        for inst in self.gasf.instances.values():
            if not self.gasf.is_confirmed(inst) or len(inst.pts) < 10:
                continue
            lo, hi = inst.robust_bounds()
            out.append(dict(id=int(inst.iid), label=self.classes[int(np.argmax(inst.probs))], confidence=float(inst.probs.max()),
                            lo=lo, hi=hi, centroid=(lo + hi) / 2, observations=int(inst.hits), views=len(inst.views)))
        return out

    def map_points(self, max_points=150000):
        P, C, W, N = self.fmap.arrays()
        if len(P) > max_points:
            idx = np.random.default_rng(0).choice(len(P), max_points, replace=False); P, C = P[idx], C[idx]
        return P, C

    def traversability(self):
        return self.occ.traversability(floor_z=self.floor_z)

    def plan_next_view(self):
        if self.planner is None:
            return None
        x, y = self.last_pose[0, 3], self.last_pose[1, 3]
        try:
            return self.planner.plan(x, y, known_free_xy=np.array(self.traj))
        except Exception as e:                       # unexplored / degenerate map early on
            return dict(error=str(e))

    def stats(self):
        ms = {k: 1000 * float(np.mean(v)) if v else 0.0 for k, v in self.timing.items()}
        return dict(frames=self.i, keyframes=len(self.timing["occ"]), detections=self.n_dets, segments=self.n_segments,
                    assets=len(self.assets()), loops=self.slam.n_loops, track_fallbacks=self.slam.n_track_fail, ms=ms)

    def finish(self, store_dir=None, dashboard_path=None, session="live", studies=(), title="GeoSemTwin-X live twin",
               traj_gt=None):
        """Pose-graph optimisation, twin commit, optional versioned store + dashboard. Returns the DigitalTwin."""
        self.slam.optimize()
        twin = DigitalTwin(title)
        twin.update(self.gasf.finalize(), self.classes, session, occ_new=self.occ)
        if store_dir:
            TwinStore(store_dir).push(twin)
        if dashboard_path:
            from .dashboard import build_dashboard
            P, C = self.map_points()
            st = self.stats()
            build_dashboard(dashboard_path, twin, P, C, traj_est=np.array(self.traj), traj_gt=traj_gt, studies=studies,
                            perf=dict(stages_ms={f"{k} (ms)": v for k, v in st["ms"].items()},
                                      details={k: v for k, v in st.items() if k != "ms"}),
                            title=title, subtitle=session)
        return twin
