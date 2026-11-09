"""Bridge to the official VIGS-SLAM implementation (https://github.com/cvg/VIGS-SLAM).

1. export_staer()  - writes a Staer walkthrough in VIGS-SLAM's input format:
       <out>/images/<timestamp_ns>.png, <out>/imu.csv (t_ns,gx,gy,gz,ax,ay,az), <out>/calib.txt (fx fy cx cy),
       <out>/staer.yaml (VIGS config with the Staer IMU noise model and camera<-IMU extrinsics Tcb)
2. run_vigs()      - runs VIGS-SLAM's demo.py on it (in VIGS-SLAM's own conda env / python)
3. load_vigs_trajectory() - reads traj_full_afterBA.txt (or the best available) and associates/interpolates
       poses to our frame timestamps, so VIGS-SLAM poses drive the GeoSemTwin-X mapping / GASF / twin.

Conventions: Staer pose frames are x-right / y-up / z-back (OpenGL); VIGS-SLAM (DROID) works in the optical
frame (x-right / y-down / z-forward). Tcb in the VIGS config maps IMU(body) points into the optical camera frame.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

GL_TO_OPT = np.diag([1.0, -1.0, -1.0, 1.0])


def _sensor_T_BS(path):
    """Parse T_BS from an EuRoC sensor.yaml without needing PyYAML."""
    txt = Path(path).read_text()
    i = txt.find("T_BS")
    j = txt.find("data:", i)
    k = txt.find("]", j)
    vals = [float(v) for v in txt[txt.find("[", j) + 1:k].replace("\n", " ").split(",")]
    return np.array(vals).reshape(4, 4)


def _sensor_value(path, key, default=None):
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith(key + ":"):
            try:
                return float(line.split(":", 1)[1].split("#")[0])
            except ValueError:
                return default
    return default


def export_staer(scene="scene03", walk="001", out_dir="data/vigs_staer", root="data/staer", scale=0.5,
                 vigs_root=None, overwrite=False):
    from huggingface_hub import hf_hub_download
    base = f"{scene}/walkthroughs/{walk}"
    files = {}
    for f in ["cam0/video.mp4", "cam0/data.csv", "cam0/sensor.yaml", "imu0/data.csv", "imu0/sensor.yaml"]:
        files[f] = hf_hub_download("staerrobotics/warehouses", f"{base}/{f}", repo_type="dataset", local_dir=root,
                                   token=True)
    out = Path(out_dir); img = out / "images"
    if overwrite and out.exists():
        shutil.rmtree(out)
    img.mkdir(parents=True, exist_ok=True)
    # frame timestamps (ns)
    ts = []
    for line in open(files["cam0/data.csv"]):
        if line.strip() and not line.startswith("#"):
            ts.append(int(float(line.split(",")[0])))
    # RGB frames
    W, H = int(1920 * scale), int(1080 * scale)
    from .data import ffmpeg_sync_args
    meta_p = out / "export_meta.json"
    old = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if old.get("decode") != "passthrough":          # exports made before the frame-sync fix are rewritten
        for f in img.glob("*.png"):
            f.unlink()
    if len(list(img.glob("*.png"))) != len(ts):
        import cv2
        cmd = ["ffmpeg", "-v", "error", "-i", files["cam0/video.mp4"], "-vf", f"scale={W}:{H}"] + ffmpeg_sync_args() + \
              ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        k = 0
        while True:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3 or k >= len(ts):
                break
            cv2.imwrite(str(img / f"{ts[k]}.png"), np.frombuffer(buf, np.uint8).reshape(H, W, 3))
            k += 1
        n_left = len(p.stdout.read()) // (W * H * 3)
        p.wait()
        print(f"wrote {k} images to {img}" + ("" if n_left == 0 else f"  (WARNING: video has {k + n_left} frames, data.csv {len(ts)})"))
    # IMU (EuRoC order t, w_xyz, a_xyz is exactly what VIGS-SLAM expects)
    rows = [l.strip() for l in open(files["imu0/data.csv"]) if l.strip() and not l.startswith("#")]
    (out / "imu.csv").write_text("\n".join(rows) + "\n")
    rate = _sensor_value(files["imu0/sensor.yaml"], "rate_hz", None)
    if rate is None and len(rows) > 2:
        t = np.array([float(r.split(",")[0]) for r in rows])
        rate = 1e9 / np.median(np.diff(t))
    # intrinsics (pinhole, no distortion in this synthetic dataset)
    fx = fy = 672.2 * scale; cx, cy = 960 * scale, 540 * scale
    intr = None
    for line in Path(files["cam0/sensor.yaml"]).read_text().splitlines():
        if line.strip().startswith("intrinsics:"):
            intr = [float(v) for v in line.split("[", 1)[1].split("]")[0].split(",")]
    if intr:
        fx, fy, cx, cy = [v * scale for v in intr[:4]]
    (out / "calib.txt").write_text(f"{fx} {fy} {cx} {cy}\n")
    # extrinsics: Tcb = T_optical<-body = (T_B_camGL @ GL_TO_OPT)^-1 @ T_B_imu
    T_B_cam = _sensor_T_BS(files["cam0/sensor.yaml"]) @ GL_TO_OPT
    T_B_imu = _sensor_T_BS(files["imu0/sensor.yaml"])
    Tcb = np.linalg.inv(T_B_cam) @ T_B_imu
    noise = {k: _sensor_value(files["imu0/sensor.yaml"], k, d) for k, d in
             (("accelerometer_noise_density", 2.0e-3), ("accelerometer_random_walk", 3.0e-3),
              ("gyroscope_noise_density", 1.7e-4), ("gyroscope_random_walk", 2.0e-5))}
    cfg = _make_config(vigs_root, rate, noise, Tcb)
    (out / "staer.yaml").write_text(cfg)
    meta = dict(scene=scene, walk=walk, frames=len(ts), decode="passthrough", imu_rate_hz=rate, intrinsics=[fx, fy, cx, cy],
                Tcb=Tcb.tolist(), timestamps_ns=[ts[0], ts[-1]])
    (out / "export_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"VIGS-SLAM input ready in {out}  ({len(ts)} frames, IMU {rate:.0f} Hz)")
    print("Tcb (optical camera <- IMU):\n", np.round(Tcb, 5))
    return out


def _make_config(vigs_root, rate, noise, Tcb):
    """Start from VIGS-SLAM's euroc.yaml (tracking / mapping settings) and replace the IMU block."""
    template = None
    if vigs_root and (Path(vigs_root) / "config" / "euroc.yaml").exists():
        template = (Path(vigs_root) / "config" / "euroc.yaml").read_text()
    imu = ["IMU:", f"  frequency: {int(round(rate))}"]
    imu += [f"  {k}: {v}" for k, v in noise.items()]
    imu += ["  imu_time_offset: 0.0", "  rgb_file_in_nanoseconds: true", "  imu_in_nanoseconds: true",
            "  init_g: [0.0, 0.0, -9.81]", "  init_bg: [0.0, 0.0, 0.0]", "  init_ba: [0.0, 0.0, 0.0]", "  Tcb_np:"]
    imu += ["    - [" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in Tcb]
    block = "\n".join(imu) + "\n"
    if template is None:
        print("NOTE: VIGS-SLAM root not given - writing only the IMU block; merge it into a copy of config/euroc.yaml")
        return block
    lines = template.splitlines(keepends=True)
    s = next(i for i, l in enumerate(lines) if l.startswith("IMU:"))
    e = next((i for i in range(s + 1, len(lines)) if lines[i].strip() and not lines[i].startswith((" ", "\t", "-"))),
             len(lines))
    return "".join(lines[:s]) + block + "\n" + "".join(lines[e:])


def vigs_command(vigs_root, export_dir, out_dir, python="python", gsmapping=False, imu_init_after=20, buffer=None,
                 stride=1):
    export_dir, out_dir = Path(export_dir).resolve(), Path(out_dir).resolve()
    n = len(list((export_dir / "images").glob("*.png")))
    cmd = [python, "demo.py", "--imagedir", str(export_dir / "images"), "--imufile", str(export_dir / "imu.csv"),
           "--calib", str(export_dir / "calib.txt"), "--config", str(export_dir / "staer.yaml"),
           "--output", str(out_dir), "--IMU_poseinit_after", str(imu_init_after), "--stride", str(stride),
           "--buffer", str(buffer or max(200, n // 5))]
    if gsmapping:
        cmd.append("--gsmapping")
    return cmd


def run_vigs(vigs_root, export_dir, out_dir, conda_env=None, python="python", gsmapping=False, **kw):
    """Runs VIGS-SLAM. conda_env='vigs-slam-5090' runs it inside that env via `conda run`."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = vigs_command(vigs_root, export_dir, out_dir, python=python, gsmapping=gsmapping, **kw)
    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd
    print("Running VIGS-SLAM:\n ", " ".join(cmd))
    with open(Path(out_dir) / "log.txt", "w") as log:
        r = subprocess.run(cmd, cwd=vigs_root, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"VIGS-SLAM failed (exit {r.returncode}); see {out_dir}/log.txt")
    return out_dir


TRAJ_PREFERENCE = ["traj_full_afterBA.txt", "traj_full_beforeBA.txt", "traj_kf_afterBA.txt", "traj_kf_beforeBA.txt",
                   "traj_kf.txt"]


def find_trajectory(out_dir):
    for f in TRAJ_PREFERENCE:
        if (Path(out_dir) / f).exists():
            return Path(out_dir) / f
    raise FileNotFoundError(f"no VIGS-SLAM trajectory in {out_dir}")


def load_vigs_trajectory(path, timestamps):
    """TUM-format file (t tx ty tz qx qy qz qw, t in s) -> 4x4 camera-to-world (optical) per timestamp,
    linearly interpolated (translation) / slerped (rotation) between trajectory samples."""
    from scipy.spatial.transform import Rotation, Slerp
    path = Path(path)
    if path.is_dir():
        path = find_trajectory(path)
    rows = np.loadtxt(path)
    rows = rows[np.argsort(rows[:, 0])]
    t = rows[:, 0]
    if t[0] > 1e12:                     # nanoseconds
        t = t / 1e9
    rot = Slerp(t, Rotation.from_quat(rows[:, 4:8]))
    ts = np.clip(np.asarray(timestamps, float), t[0], t[-1])
    R = rot(ts).as_matrix()
    P = np.stack([np.interp(ts, t, rows[:, k]) for k in (1, 2, 3)], 1)
    out = []
    for k in range(len(ts)):
        T = np.eye(4); T[:3, :3] = R[k]; T[:3, 3] = P[k]; out.append(T)
    return out, path.name
