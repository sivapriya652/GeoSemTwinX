"""Dataset adapters. Every sequence exposes the same interface:

    seq.K            3x3 intrinsics
    len(seq)         number of frames
    seq.frame(i)     dict(rgb uint8 HxWx3, depth float32 metres, [instance int32], [semantic int32])
    seq.gt_poses     list of 4x4 camera-to-world (optical frame) or None
    seq.timestamps   seconds
    seq.instance_class  {instance id: class name} for GT evaluation (optional)
"""
import json
import os
import re
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np


_SYNC_ARGS = None


def ffmpeg_sync_args(mode="passthrough"):
    """Decode every stored frame exactly once, in stream order. Without this, ffmpeg's rawvideo output runs in
    constant-frame-rate mode and duplicates / drops frames whenever the container timestamps are uneven, so
    decoded frame k is no longer frame k of data.csv / poses.csv (and RGB, depth and instance streams drift apart)."""
    global _SYNC_ARGS
    if mode != "passthrough":
        return []
    if _SYNC_ARGS is None:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "nullsrc=d=0.1", "-fps_mode", "passthrough",
                            "-f", "null", "-"], capture_output=True)
        _SYNC_ARGS = ["-fps_mode", "passthrough"] if r.returncode == 0 else ["-vsync", "0"]
    return list(_SYNC_ARGS)


def quat_to_R(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# --------------------------------------------------------------------------- synthetic
class SyntheticSequence:
    def __init__(self, session="A", n_frames=360, stride=1, seed=0, scale=1.0):
        from .synthetic import build_sequence
        self.sim, poses, self.instance_class, self.changes = build_sequence(session, n_frames, seed, scale)
        self.idx = list(range(0, n_frames, stride))
        self.gt_poses = [poses[i] for i in self.idx]
        self.K = self.sim.K.copy()
        self.timestamps = [i / 30.0 for i in self.idx]
        self.instance_names = self.sim.instance_names
        self._cache = {}

    def __len__(self):
        return len(self.idx)

    def frame(self, i):
        if i not in self._cache:
            self._cache[i] = self.sim.render(self.gt_poses[i])
        return self._cache[i]


# --------------------------------------------------------------------------- TUM RGB-D
TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _read_tum_list(path):
    out = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        out.append((float(p[0]), p[1:]))
    return out


def _associate(a, b, max_dt=0.02):
    tb = np.array([t for t, _ in b])
    pairs, used = [], set()
    for ta, va in a:
        j = int(np.argmin(np.abs(tb - ta)))
        if abs(tb[j] - ta) < max_dt and j not in used:
            used.add(j)
            pairs.append((ta, va, b[j][1]))
    return pairs


class TUMSequence:
    """name e.g. 'rgbd_dataset_freiburg1_desk'. Downloads (~0.4 GB) on first use."""
    URL = "https://cvg.cit.tum.de/rgbd/dataset/{fr}/{name}.tgz"

    def __init__(self, name="rgbd_dataset_freiburg1_desk", root="data/tum", stride=1, max_frames=None):
        fr = re.search(r"freiburg\d", name).group(0)
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        seq_dir = root / name
        if not seq_dir.exists():
            tgz = root / f"{name}.tgz"
            if not tgz.exists():
                print("Downloading", name, "...")
                urllib.request.urlretrieve(self.URL.format(fr=fr, name=name), tgz)
            with tarfile.open(tgz) as tf:
                tf.extractall(root)
        fx, fy, cx, cy = TUM_INTRINSICS[fr]
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        rgb = _read_tum_list(seq_dir / "rgb.txt")
        depth = _read_tum_list(seq_dir / "depth.txt")
        pairs = _associate(rgb, depth)
        gt = _read_tum_list(seq_dir / "groundtruth.txt")
        gt_t = np.array([t for t, _ in gt])
        frames, poses = [], []
        for t, vr, vd in pairs:
            j = int(np.argmin(np.abs(gt_t - t)))
            if abs(gt_t[j] - t) > 0.02:
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, gt[j][1])
            T = np.eye(4); T[:3, :3] = quat_to_R(qw, qx, qy, qz); T[:3, 3] = (tx, ty, tz)
            frames.append((t, seq_dir / vr[0], seq_dir / vd[0])); poses.append(T)
        frames, poses = frames[::stride], poses[::stride]
        if max_frames:
            frames, poses = frames[:max_frames], poses[:max_frames]
        self.files, self.gt_poses = frames, poses
        self.timestamps = [f[0] for f in frames]
        self.instance_class = None

    def __len__(self):
        return len(self.files)

    def frame(self, i):
        _, fr, fd = self.files[i]
        rgb = cv2.cvtColor(cv2.imread(str(fr)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(fd), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        return dict(rgb=rgb, depth=depth)


# --------------------------------------------------------------------------- Staer warehouses
STAER_CLASS_RULES = [            # keyword in prim path -> class (first match wins)
    ("pallettruck", "pallet truck"), ("forklift", "forklift"),
    ("cardbox", "cardboard box"), ("carton", "cardboard box"), ("box", "cardboard box"),
    ("palette", "pallet"), ("pallet", "pallet"),
    ("rackshelf", "rack"), ("rack", "rack"), ("barrel", "barrel"), ("drum", "barrel"),
]


def staer_parent(prim):
    if not isinstance(prim, str) or prim.upper() == "INVALID":
        return None
    m = re.search(r"(/World/[^/]+/Prim_\d+)", prim)
    return m.group(1) if m else None


def staer_class(prim):
    p = prim.lower().replace("_", "")
    for key, cls in STAER_CLASS_RULES:
        if key in p:
            return cls
    return None


class StaerSequence:
    """Staer Robotics warehouses (gated on Hugging Face: log in first).

    Camera convention verified for this dataset: pose camera frame is x-right,
    y-up, z-backward, so optical->camera is diag(1,-1,-1). Depth is planar.
    """
    FILES = ["cam0/video.mp4", "cam0/poses.csv", "depth0/video.mkv", "depth0/depth_ranges.json",
             "instance0/video.mkv", "instance0/id_to_labels.json"]

    def __init__(self, scene="scene03", walk="001", root="data/staer", scale=0.5,
                 stride=2, max_frames=None, start=0, sync="passthrough"):
        from huggingface_hub import hf_hub_download
        self.sync = sync
        base = f"{scene}/walkthroughs/{walk}"
        paths = {}
        for f in self.FILES:
            paths[f] = hf_hub_download("staerrobotics/warehouses", f"{base}/{f}", repo_type="dataset",
                                       local_dir=root, token=True)
        try:                                     # true frame timestamps (needed to align external trajectories)
            paths["cam0/data.csv"] = hf_hub_download("staerrobotics/warehouses", f"{base}/cam0/data.csv",
                                                     repo_type="dataset", local_dir=root, token=True)
        except Exception:
            paths["cam0/data.csv"] = None
        try:                                     # the depth stream has its own pose file (sensor offset)
            paths["depth0/poses.csv"] = hf_hub_download("staerrobotics/warehouses", f"{base}/depth0/poses.csv",
                                                        repo_type="dataset", local_dir=root, token=True)
        except Exception:
            paths["depth0/poses.csv"] = None
        W, H = int(1920 * scale), int(1080 * scale)
        self.K = np.array([[672.2 * scale, 0, 960 * scale], [0, 672.2 * scale, 540 * scale], [0, 0, 1]])
        ranges = json.load(open(paths["depth0/depth_ranges.json"]))
        flip = np.diag([1., -1., -1.])

        def read_poses(path):
            out = []
            for l in open(path):
                if not l.strip() or l.startswith("#"):
                    continue
                p = [float(v) for v in l.split(",")]
                T = np.eye(4); T[:3, :3] = quat_to_R(p[4], p[5], p[6], p[7]) @ flip; T[:3, 3] = p[1:4]
                out.append(T)
            return out
        cam_poses = read_poses(paths["cam0/poses.csv"])
        poses_all = cam_poses
        self.depth_pose_offset_m = 0.0
        if paths["depth0/poses.csv"]:
            dep = read_poses(paths["depth0/poses.csv"])
            if len(dep) == len(cam_poses):
                self.depth_pose_offset_m = float(np.median([np.linalg.norm(a[:3, 3] - b[:3, 3])
                                                            for a, b in zip(dep, cam_poses)]))
                poses_all = dep            # depth pixels live in the depth sensor frame
        print(f"Staer: cam0 <-> depth0 pose offset = {100 * self.depth_pose_offset_m:.1f} cm "
              f"({'using depth0 poses for geometry' if poses_all is not cam_poses else 'using cam0 poses'})")
        n = len(poses_all)
        self.idx = list(range(start, n, stride))[: max_frames or None]
        sel = set(self.idx)
        self.decoded_frames = {}
        self._stop_early, self._last_sel = max_frames is not None, max(sel)
        rgb = self._decode(paths["cam0/video.mp4"], W, H, "rgb24", 3, sel, "bilinear")
        inv = self._decode(paths["depth0/video.mkv"], W, H, "gray16le", 1, sel, "neighbor")
        inst = self._decode(paths["instance0/video.mkv"], W, H, "gray16le", 1, sel, "neighbor")
        bad = {k: v for k, v in self.decoded_frames.items() if v != n}
        if bad:
            print(f"WARNING: decoded frame counts {bad} != {n} poses - streams may be misaligned")
        m = min(len(rgb), len(inv), len(inst))
        if m < len(self.idx):
            self.idx = self.idx[:m]
        self._rgb, self._inst = rgb, inst
        self._depth = []
        for k, i in enumerate(self.idx):
            v = inv[k].astype(np.float32); r = ranges[i]
            iv = r["inv_min"] + (v - 1) / 65534.0 * (r["inv_max"] - r["inv_min"])
            d = np.where(v > 0, 1.0 / np.maximum(iv, 1e-6), 0).astype(np.float32)
            self._depth.append(d)
        self.gt_poses = [poses_all[i] for i in self.idx]
        stamps = None
        if paths.get("cam0/data.csv"):
            stamps = [float(l.split(",")[0]) / 1e9 for l in open(paths["cam0/data.csv"]) if l.strip() and not l.startswith("#")]
        self.timestamps = [stamps[i] for i in self.idx] if stamps and len(stamps) >= n else [i / 20.0 for i in self.idx]
        # instance id -> parent asset + class
        labels = json.load(open(paths["instance0/id_to_labels.json"]))
        parents, self.parent_class = {}, {}
        for k, prim in labels.items():
            par = staer_parent(prim)
            if par is None:
                continue
            cls = staer_class(prim)
            parents[int(k)] = par
            if cls and par not in self.parent_class:
                self.parent_class[par] = cls
        names = sorted(set(parents.values()))
        pid = {n: i + 1 for i, n in enumerate(names)}
        lut = np.zeros(65536, np.int32)
        for k, par in parents.items():
            if k < 65536 and par in self.parent_class:      # keep only classified assets
                lut[k] = pid[par]
        self._lut = lut
        self.instance_names = ["background"] + names
        self.instance_class = {pid[n]: self.parent_class[n] for n in names if n in self.parent_class}

    def _decode(self, path, W, H, pix, ch, sel, flags):
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}:flags={flags}"] + ffmpeg_sync_args(self.sync) + \
              ["-f", "rawvideo", "-pix_fmt", pix, "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        dtype = np.uint16 if pix == "gray16le" else np.uint8
        nbytes = W * H * ch * np.dtype(dtype).itemsize
        out, k = [], 0
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            if k in sel:
                a = np.frombuffer(buf, dtype).reshape(H, W, ch) if ch > 1 else np.frombuffer(buf, dtype).reshape(H, W)
                out.append(a.copy())
            k += 1
            if self._stop_early and k > self._last_sel:
                proc.kill()
                break
        proc.wait()
        if not self._stop_early:
            self.decoded_frames[Path(path).parent.name] = k
        return out

    def __len__(self):
        return len(self.idx)

    def frame(self, i):
        return dict(rgb=self._rgb[i], depth=self._depth[i], instance=self._lut[self._inst[i]])
