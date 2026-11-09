"""Dense mapping: uncertainty-weighted fused surface map, log-odds occupancy + uncertainty, traversability."""
import numpy as np
import open3d as o3d



def depth_sigma(z):
    """Structured-light / ToF style axial noise model (metres)."""
    return 0.0012 + 0.0019 * (np.maximum(z, 0.4) - 0.4) ** 2


def backproject(depth, K, T_wc, stride=1, mask=None, max_depth=None):
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    m = z > 0
    if max_depth:
        m &= z < max_depth
    if mask is not None:
        m &= mask[::stride, ::stride]
    x = (u[m] - K[0, 2]) / K[0, 0] * z[m]
    y = (v[m] - K[1, 2]) / K[1, 1] * z[m]
    P = np.stack([x, y, z[m]], 1)
    return P @ T_wc[:3, :3].T + T_wc[:3, 3], z[m], (v[m], u[m])


class FusedMap:
    """Uncertainty-weighted voxel fusion map (surfel-style), pure NumPy.

    Every depth pixel is back-projected and accumulated into a voxel hash with weight
    w = 1/sigma_z(z)^2 * exp(-sigma_pose / 0.1): far, noisy depth and badly-tracked frames count less.
    Frames whose pose sigma exceeds a gate are skipped and depth is cut where the sensor noise exceeds
    sigma_max. Each voxel keeps the weighted mean position and colour, the total weight and the number
    of observations; voxels seen fewer than `min_obs` times are dropped as noise when exporting.
    (Replaces Open3D's ScalableTSDFVolume, which returns empty surfaces in the Open3D 0.20 / Python 3.13
    build used on Colab.)"""

    def __init__(self, K, width, height, voxel=0.02, depth_max=4.0, sigma_max=0.03, pose_sigma_gate=0.15,
                 stride=None, min_obs=2, uncertainty_weighting=True):
        self.K = np.asarray(K, float)
        self.voxel = float(voxel)
        zs = np.linspace(0.4, 25, 4000)
        self.depth_max = float(min(depth_max, zs[depth_sigma(zs) < sigma_max].max()))
        self.gate = pose_sigma_gate
        # sample roughly 2 pixels per voxel footprint at mid range
        self.stride = stride or max(1, int(K[0, 0] * voxel / max(self.depth_max / 2, 0.5) / 2))
        self.min_obs = min_obs
        self.weighting = uncertainty_weighting          # False = plain averaging (ablation)
        self.n_integrated = self.n_skipped = 0
        self._buf = []
        self.keys = np.empty(0, np.int64); self.acc = np.empty((0, 8))      # xyz*w, rgb*w, w, count

    def integrate(self, frame, T_wc, pose_sigma=0.0):
        if pose_sigma > self.gate:
            self.n_skipped += 1
            return
        P, z, (v, u) = backproject(frame["depth"], self.K, T_wc, stride=self.stride, max_depth=self.depth_max)
        if len(P) == 0:
            return
        if self.weighting:
            w = (1.0 / depth_sigma(z) ** 2) * np.exp(-pose_sigma / 0.1)
            w = w / w.max()
        else:
            w = np.ones_like(z)
        col = frame["rgb"][v, u].astype(np.float64) / 255.0
        ijk = np.floor(P / self.voxel).astype(np.int64) + (1 << 20)
        k = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        self._buf.append((k, np.column_stack([P * w[:, None], col * w[:, None], w, np.ones_like(w)])))
        self.n_integrated += 1
        if sum(len(b[0]) for b in self._buf) > 1_500_000:
            self._consolidate()

    def _consolidate(self):
        if not self._buf:
            return
        keys = np.concatenate([self.keys] + [b[0] for b in self._buf])
        acc = np.concatenate([self.acc] + [b[1] for b in self._buf])
        self._buf = []
        u, inv = np.unique(keys, return_inverse=True)
        out = np.zeros((len(u), acc.shape[1]))
        np.add.at(out, inv, acc)
        self.keys, self.acc = u, out

    def arrays(self, min_obs=None):
        self._consolidate()
        m = self.acc[:, 7] >= (min_obs or self.min_obs)
        a = self.acc[m]
        w = a[:, 6:7]
        return a[:, 0:3] / w, np.clip(a[:, 3:6] / w, 0, 1), a[:, 6], a[:, 7].astype(int)

    def point_cloud(self):
        P, C, _, _ = self.arrays()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        pc.colors = o3d.utility.Vector3dVector(C)
        return pc


TSDFMap = FusedMap          # backwards-compatible name


class VoxelGrid:
    """Dense voxel grid helper (index <-> world)."""

    def __init__(self, bounds_min, bounds_max, res):
        self.res = float(res)
        self.min = np.asarray(bounds_min, float)
        self.shape = np.ceil((np.asarray(bounds_max, float) - self.min) / res).astype(int)

    def index(self, P):
        ijk = np.floor((P - self.min) / self.res).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < self.shape), axis=1)
        return ijk, ok

    def flat(self, ijk):
        return (ijk[:, 0] * self.shape[1] + ijk[:, 1]) * self.shape[2] + ijk[:, 2]

    def center(self, ijk):
        return self.min + (ijk + 0.5) * self.res


class OccupancyMap(VoxelGrid):
    """Log-odds occupancy + per-voxel reconstruction uncertainty (fused depth sigma)."""
    L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5

    def __init__(self, bounds_min, bounds_max, res=0.1, max_range=6.0):
        super().__init__(bounds_min, bounds_max, res)
        n = int(np.prod(self.shape))
        self.logodds = np.zeros(n, np.float32)
        self.inv_var = np.zeros(n, np.float32)      # sum of 1/sigma^2 of surface observations
        self.hits = np.zeros(n, np.uint16)
        self.max_range = max_range

    def integrate(self, depth, K, T_wc, stride=8, pose_sigma=0.0):
        P, z, _ = backproject(depth, K, T_wc, stride=stride, max_depth=self.max_range)
        self.integrate_points(P, T_wc[:3, 3], z, pose_sigma)

    def integrate_points(self, P, origin, z=None, pose_sigma=0.0):
        if len(P) == 0:
            return
        d = P - origin
        L = np.linalg.norm(d, axis=1)
        steps = np.arange(0.25, self.max_range, self.res * 0.9)
        # free space samples along each ray, stopping one voxel before the hit
        frac = steps[None, :] / np.maximum(L[:, None], 1e-6)
        keep = steps[None, :] < (L[:, None] - self.res)
        S = origin + d[:, None, :] * frac[..., None]
        S = S[keep]
        ijk, ok = self.index(S)
        free = np.unique(self.flat(ijk[ok]))
        ijk_h, ok_h = self.index(P)
        hit = self.flat(ijk_h[ok_h])
        hit_u, cnt = np.unique(hit, return_counts=True)
        free = np.setdiff1d(free, hit_u, assume_unique=True)
        self.logodds[free] = np.clip(self.logodds[free] + self.L_FREE, self.L_MIN, self.L_MAX)
        self.logodds[hit_u] = np.clip(self.logodds[hit_u] + self.L_OCC, self.L_MIN, self.L_MAX)
        sig = depth_sigma(z if z is not None else L)[ok_h] + pose_sigma
        np.add.at(self.inv_var, hit, (1.0 / np.maximum(sig, 1e-3) ** 2).astype(np.float32))
        self.hits[hit_u] = np.minimum(self.hits[hit_u].astype(np.int32) + cnt, 65000).astype(np.uint16)

    # states: -1 unknown, 0 free, 1 occupied
    def state(self):
        s = np.full(self.logodds.shape, -1, np.int8)
        s[self.logodds < -0.3] = 0
        s[self.logodds > 0.6] = 1
        return s

    def surface_sigma(self):
        with np.errstate(divide="ignore"):
            return np.where(self.inv_var > 0, 1.0 / np.sqrt(self.inv_var), np.inf)

    def coverage(self, gt_surface_flat):
        st = self.state()
        return float((st[gt_surface_flat] == 1).mean())

    def traversability(self, floor_z=0.0, robot_height=1.8, clearance=0.15):
        """2D grid: 1 traversable, 0 obstacle, -1 unknown."""
        st = self.state().reshape(self.shape)
        kz0 = int(np.floor((floor_z + clearance - self.min[2]) / self.res))
        kz1 = int(np.ceil((floor_z + robot_height - self.min[2]) / self.res))
        kz0, kz1 = max(kz0, 0), min(kz1, self.shape[2])
        band = st[:, :, kz0:kz1]
        occ = (band == 1).any(axis=2)
        known_free = (band == 0).sum(axis=2) >= 1
        grid = np.full(occ.shape, -1, np.int8)
        grid[known_free] = 1
        grid[occ] = 0
        return grid
