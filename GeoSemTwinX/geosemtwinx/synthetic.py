"""Synthetic warehouse RGB-D generator (CPU, Open3D ray casting).

Produces a sequence with ground-truth camera poses, depth, RGB (procedural
texture so photometric odometry has gradients), per-pixel instance and semantic
labels, plus an optional "session B" with known asset changes for evaluating
change detection.  Used to test every non-neural module of GeoSemTwin-X with
exact ground truth.
"""
import cv2
import numpy as np
import open3d as o3d

CLASSES = ["floor", "wall", "rack", "pallet", "cardboard box", "forklift", "barrel"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
ASSET_CLASSES = ["pallet", "cardboard box", "forklift", "barrel"]

BASE_COLOR = {
    "floor": (0.55, 0.55, 0.52), "wall": (0.80, 0.78, 0.72), "rack": (0.15, 0.30, 0.65),
    "pallet": (0.62, 0.45, 0.25), "cardboard box": (0.72, 0.55, 0.33),
    "forklift": (0.90, 0.70, 0.05), "barrel": (0.20, 0.45, 0.25),
}


def _box(size, center, yaw=0.0):
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(-np.asarray(size) / 2.0)
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    m.rotate(R, center=(0, 0, 0))
    m.translate(center)
    return m


def _cyl(radius, height, center):
    m = o3d.geometry.TriangleMesh.create_cylinder(radius, height, resolution=24)
    m.translate(center)
    return m


def _merge(parts):
    out = o3d.geometry.TriangleMesh()
    for p in parts:
        out += p
    return out


def make_forklift(center, yaw):
    cx, cy, _ = center
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    parts = [
        _box((1.1, 1.9, 1.0), (0, 0, 0.55)),          # body
        _box((1.0, 0.6, 1.1), (0, -0.55, 1.6)),       # cabin roof block
        _box((0.9, 0.12, 2.2), (0, 1.0, 1.1)),        # mast
        _box((0.12, 1.0, 0.06), (-0.3, 1.5, 0.08)),   # fork L
        _box((0.12, 1.0, 0.06), (0.3, 1.5, 0.08)),    # fork R
    ]
    m = _merge(parts)
    m.rotate(R, center=(0, 0, 0))
    m.translate((cx, cy, 0))
    return m


def make_rack(x0, y0, length, depth=1.0, levels=3, height=4.0, along_x=True):
    parts = []
    n_up = int(length // 2.0) + 1
    for k in range(n_up):
        for d in (0, depth):
            if along_x:
                c = (x0 + k * (length / (n_up - 1)), y0 + d, height / 2)
            else:
                c = (x0 + d, y0 + k * (length / (n_up - 1)), height / 2)
            parts.append(_box((0.08, 0.08, height), c))
    for lv in range(levels):
        z = 0.3 + lv * (height - 0.4) / (levels - 1) if levels > 1 else 0.3
        if along_x:
            parts.append(_box((length, depth, 0.05), (x0 + length / 2, y0 + depth / 2, z)))
        else:
            parts.append(_box((depth, length, 0.05), (x0 + depth / 2, y0 + length / 2, z)))
    return _merge(parts)


def default_layout(seed=0):
    """List of objects: dict(name, cls, mesh_fn args). Assets have stable names."""
    rng = np.random.default_rng(seed)
    objs = []
    W, D, H = 24.0, 16.0, 6.0
    objs.append(dict(name="floor", cls="floor", mesh=_box((W, D, 0.02), (W / 2, D / 2, -0.01))))
    objs.append(dict(name="ceiling", cls="wall", mesh=_box((W, D, 0.1), (W / 2, D / 2, H + 0.05))))
    objs.append(dict(name="wall_s", cls="wall", mesh=_box((W, 0.1, H), (W / 2, -0.05, H / 2))))
    objs.append(dict(name="wall_n", cls="wall", mesh=_box((W, 0.1, H), (W / 2, D + 0.05, H / 2))))
    objs.append(dict(name="wall_w", cls="wall", mesh=_box((0.1, D, H), (-0.05, D / 2, H / 2))))
    objs.append(dict(name="wall_e", cls="wall", mesh=_box((0.1, D, H), (W + 0.05, D / 2, H / 2))))
    # racks along north and south walls and a middle row
    objs.append(dict(name="rack_1", cls="rack", mesh=make_rack(2, 14.4, 20)))
    objs.append(dict(name="rack_2", cls="rack", mesh=make_rack(2, 0.6, 20)))
    objs.append(dict(name="rack_3", cls="rack", mesh=make_rack(4, 7.5, 12)))
    # pallets with boxes, forklifts, barrels in aisles
    pallet_spots = [(3.0, 4.0), (6.0, 3.2), (9.5, 4.2), (13.0, 3.5), (17.0, 4.0), (20.5, 3.4),
                    (3.5, 12.0), (8.0, 12.4), (12.5, 11.8), (16.5, 12.3), (21.0, 11.9)]
    for i, (x, y) in enumerate(pallet_spots):
        yaw = rng.uniform(-0.3, 0.3)
        objs.append(dict(name=f"pallet_{i:02d}", cls="pallet", mesh=_box((1.2, 1.0, 0.15), (x, y, 0.075), yaw)))
        if i % 2 == 0:
            objs.append(dict(name=f"box_{i:02d}", cls="cardboard box",
                             mesh=_box((0.6, 0.5, 0.5), (x, y, 0.15 + 0.25), yaw)))
    for i, (x, y, yaw) in enumerate([(11.0, 5.6, 0.4), (19.0, 10.0, -1.2), (5.5, 9.8, 2.0)]):
        objs.append(dict(name=f"forklift_{i:02d}", cls="forklift", mesh=make_forklift((x, y, 0), yaw)))
    for i, (x, y) in enumerate([(22.5, 6.0), (22.4, 9.5), (1.5, 6.5), (14.5, 9.6)]):
        objs.append(dict(name=f"barrel_{i:02d}", cls="barrel", mesh=_cyl(0.3, 0.9, (x, y, 0.0))))
    return objs


def apply_changes(objs, seed=1):
    """Session B: move, remove and add assets. Returns new list + GT change list."""
    out, changes = [], []
    for o in objs:
        if o["name"] == "pallet_07":                    # removed
            changes.append(dict(name="pallet_07", type="removed"))
            continue
        if o["name"] == "pallet_03":                    # moved 1.6 m
            m = o3d.geometry.TriangleMesh(o["mesh"]); m.translate((1.6, 0.0, 0))
            out.append(dict(o, mesh=m)); changes.append(dict(name="pallet_03", type="moved"))
            continue
        if o["name"] == "forklift_01":                  # moved and rotated
            m = o3d.geometry.TriangleMesh(o["mesh"])
            c = m.get_center()
            m.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0.9)), center=c)
            m.translate((-2.0, 0.6, 0))
            out.append(dict(o, mesh=m)); changes.append(dict(name="forklift_01", type="moved"))
            continue
        if o["name"] == "barrel_03":                    # removed
            changes.append(dict(name="barrel_03", type="removed"))
            continue
        out.append(o)
    out.append(dict(name="barrel_new", cls="barrel", mesh=_cyl(0.3, 0.9, (10.5, 2.6, 0.0))))
    changes.append(dict(name="barrel_new", type="added"))
    out.append(dict(name="box_new", cls="cardboard box", mesh=_box((0.6, 0.5, 0.5), (15.0, 5.0, 0.25), 0.2)))
    changes.append(dict(name="box_new", type="added"))
    return out, changes


def look_at(eye, target, up=(0, 0, 1)):
    """Camera-to-world pose, optical frame (x right, y down, z forward)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye; z = z / np.linalg.norm(z)
    x = np.cross(z, up); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def default_trajectory(n=360, height=1.4, loop_frames=720):
    """Loop around the middle rack row (one loop = loop_frames frames, ~0.055 m/frame at 30 fps),
    looking ahead with gentle head turns. Returns the first n poses."""
    t = np.arange(n) * 2 * np.pi / loop_frames
    cx, cy, ax, ay = 12.0, 8.0, 8.6, 3.3
    poses = []
    for k, tt in enumerate(t):
        eye = np.array([cx + ax * np.cos(tt), cy + ay * np.sin(tt), height + 0.05 * np.sin(5 * tt)])
        ahead = np.array([cx + ax * np.cos(tt + 0.25), cy + ay * np.sin(tt + 0.25), 0.9])
        d = ahead - eye
        side = np.cross([0, 0, 1], d / np.linalg.norm(d))
        target = ahead + 2.5 * np.sin(3 * tt) * side   # look left/right
        poses.append(look_at(eye, target))
    return poses


class SyntheticWarehouse:
    def __init__(self, objs, width=640, height=480, fx=525.0, fy=525.0, cx=319.5, cy=239.5,
                 depth_noise=True, max_depth=20.0, seed=0):
        self.objs, self.W, self.H = objs, width, height
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        self.depth_noise, self.max_depth = depth_noise, max_depth
        self.rng = np.random.default_rng(seed)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.gid_to_obj = {}
        self.instance_names = ["background"]
        for o in objs:
            gid = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o["mesh"]))
            self.gid_to_obj[gid] = o
            if o["cls"] in ASSET_CLASSES:
                self.instance_names.append(o["name"])
        self.inst_id = {n: i for i, n in enumerate(self.instance_names)}
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        self.dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1).reshape(-1, 3)

    def render(self, T_wc):
        R, t = T_wc[:3, :3], T_wc[:3, 3]
        dirs = self.dirs_cam @ R.T
        rays = np.concatenate([np.repeat(t[None], len(dirs), 0), dirs], 1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        z = ans["t_hit"].numpy().reshape(self.H, self.W)       # planar depth (dir has z=1)
        gid = ans["geometry_ids"].numpy().reshape(self.H, self.W)
        normals = ans["primitive_normals"].numpy().reshape(self.H, self.W, 3)
        valid = np.isfinite(z) & (z < self.max_depth)
        pts = (t + dirs.reshape(self.H, self.W, 3) * np.where(valid, z, 0)[..., None])
        sem = np.zeros((self.H, self.W), np.int32)
        inst = np.zeros((self.H, self.W), np.int32)
        base = np.zeros((self.H, self.W, 3))
        for g, o in self.gid_to_obj.items():
            m = (gid == g) & valid
            if not m.any():
                continue
            sem[m] = CLASS_ID[o["cls"]]
            if o["cls"] in ASSET_CLASSES:
                inst[m] = self.inst_id[o["name"]]
            base[m] = BASE_COLOR[o["cls"]]
        # procedural texture: multi-frequency pattern on world coords + per-object tint
        p = pts
        tex = (0.55 + 0.18 * np.sin(3.1 * p[..., 0] + 1.7 * p[..., 2]) * np.sin(2.3 * p[..., 1] - 0.9 * p[..., 2])
               + 0.12 * np.sin(11.0 * p[..., 0] + 7.0 * p[..., 1] + 5.0 * p[..., 2])
               + 0.10 * ((np.floor(p[..., 0] * 2) + np.floor(p[..., 1] * 2) + np.floor(p[..., 2] * 2)) % 2))
        # high-frequency surface detail (hashed value noise at 4 cm and 12 cm) -> trackable corners
        for cell, amp in ((25.0, 0.22), (8.0, 0.18)):
            q = np.floor(p * cell)
            hsh = np.sin(q[..., 0] * 12.9898 + q[..., 1] * 78.233 + q[..., 2] * 37.719) * 43758.5453
            tex = tex + amp * (hsh - np.floor(hsh) - 0.5)
        light = 0.55 + 0.45 * np.clip(np.abs(normals @ np.array([0.3, 0.2, 0.93])), 0, 1)
        rgb = np.clip(base * tex[..., None] * light[..., None] * 1.25, 0, 1)
        rgb[~valid] = 0
        rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 0.8)          # lens blur / anti-aliasing
        rgb = (rgb * 255 + self.rng.normal(0, 2.0, rgb.shape)).clip(0, 255).astype(np.uint8)
        depth = np.where(valid, z, 0.0).astype(np.float32)
        if self.depth_noise:
            sigma = 0.0012 + 0.0019 * (np.maximum(depth, 0.4) - 0.4) ** 2
            depth = depth + self.rng.normal(0, 1, depth.shape) * sigma
            depth[self.rng.random(depth.shape) < 0.01] = 0.0
            depth[~valid] = 0.0
            depth = np.round(depth * 1000) / 1000.0                      # mm quantisation
        return dict(rgb=rgb, depth=depth.astype(np.float32), semantic=sem, instance=inst)


def build_sequence(session="A", n_frames=360, seed=0, scale=1.0):
    objs = default_layout(seed)
    changes = []
    if session == "B":
        objs, changes = apply_changes(objs)
    poses = default_trajectory(n_frames)
    sim = SyntheticWarehouse(objs, width=int(640 * scale), height=int(480 * scale), fx=525.0 * scale,
                             fy=525.0 * scale, cx=320 * scale - 0.5, cy=240 * scale - 0.5,
                             seed=seed + (7 if session == "B" else 0))
    inst_class = {sim.inst_id[o["name"]]: o["cls"] for o in objs if o["cls"] in ASSET_CLASSES}
    return sim, poses, inst_class, changes
