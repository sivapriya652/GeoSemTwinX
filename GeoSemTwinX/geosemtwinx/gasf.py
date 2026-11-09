"""Geometry-Aware Semantic Fusion (GASF) and baselines.

A 2D detection (label, score, mask) is lifted to a 3D *segment* with the
estimated camera pose. Segments are associated to persistent 3D *instances*
with a score that combines

    semantic agreement      (fused class distribution, cosine)
    voxel overlap           (3D support overlap, 0.2 m voxels)
    centroid proximity      (size-adaptive Gaussian)
    geometric descriptor    (log extents, PCA shape, surface-normal histogram)

Matching is one-to-one (Hungarian). Labels are fused with confidence weights
(detector score x mask score x depth quality x pose confidence). Instances are
confirmed only after multi-view agreement, stale false positives are removed by
visibility-checked misses, and duplicates are merged by 3D overlap.
"""
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .mapping import backproject, depth_sigma

OFF = 1 << 20


def vkeys(P, res):
    ijk = np.floor(P / res).astype(np.int64) + OFF
    return np.unique((ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2])


_NB = np.array([(a << 42) | (b << 21) | c for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)


def dilate_keys(keys):
    """26-neighbourhood dilation of encoded voxel keys."""
    return np.unique((keys[:, None] + _NB[None, :]).ravel())


def keys_to_centers(keys, res):
    i = (keys >> 42) - OFF; j = ((keys >> 21) & (2 ** 21 - 1)) - OFF; k = (keys & (2 ** 21 - 1)) - OFF
    return (np.stack([i, j, k], 1) + 0.5) * res


def largest_near_component(P, z, res=0.1):
    """Voxel connected components (26-nbhd); keep the biggest component that is not
    behind the object (rejects mask bleed onto background)."""
    if len(P) < 30:
        return np.ones(len(P), bool)
    ijk = np.floor((P - P.min(0)) / res).astype(np.int64)
    shape = ijk.max(0) + 1
    if np.prod(shape) > 4e6:                                   # pathological mask, fall back to depth gate
        return z < np.median(z) + 0.5
    grid = np.zeros(shape, bool); grid[tuple(ijk.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    pl = lab[tuple(ijk.T)]
    zmed = np.percentile(z, 40)
    best, best_n = 1, -1
    for c in range(1, n + 1):
        m = pl == c
        cnt = m.sum()
        if cnt > best_n and np.median(z[m]) <= zmed + 0.6:
            best, best_n = c, cnt
    return pl == best


def descriptor(P):
    ext = np.sort(np.ptp(P, 0))[::-1] + 1e-3
    C = np.cov((P - P.mean(0)).T) + 1e-9 * np.eye(3)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    l1, l2, l3 = ev / ev.sum()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[:: max(1, len(P) // 1500)]))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(16))
    nz = np.abs(np.asarray(pc.normals)[:, 2])
    hist = np.histogram(nz, bins=4, range=(0, 1))[0] / max(len(nz), 1)
    return np.concatenate([np.log(ext), [(l1 - l2) / l1, (l2 - l3) / l1, l3 / l1], 2 * hist])


class Segment:
    __slots__ = ("sid", "frame", "label", "score", "probs", "pts", "centroid", "extent", "vox",
                 "desc", "weight", "cam", "gt", "gt_purity", "npix")


def make_segments(frame, det_list, K, T_wc, classes, frame_idx, pose_sigma=0.0, max_depth=8.0,
                  gt_instance=None, start_sid=0, min_points=60, floor_classes=(), floor_z=0.0, floor_gap=0.6,
                  min_score=0.0):
    """floor_classes: classes that must touch the floor (e.g. forklift, pallet truck). A detection of
    such a class whose 3D support starts > floor_gap above the floor is rejected (geometric sanity check)."""
    cls_index = {c: i for i, c in enumerate(classes)}
    segs = []
    for d in det_list:
        if d["label"] not in cls_index or d["score"] < min_score:
            continue
        P, z, (vv, uu) = backproject(frame["depth"], K, T_wc, mask=d["mask"], max_depth=max_depth)
        if len(P) < min_points:
            continue
        keep = largest_near_component(P, z)
        P, z, vv, uu = P[keep], z[keep], vv[keep], uu[keep]
        if len(P) < min_points:
            continue
        if d["label"] in floor_classes and np.percentile(P[:, 2], 2) > floor_z + floor_gap:
            continue
        s = Segment()
        s.sid, s.frame, s.label, s.score = start_sid + len(segs), frame_idx, d["label"], float(d["score"])
        s.probs = np.zeros(len(classes)); s.probs[cls_index[d["label"]]] = 1.0
        sub = P[:: max(1, len(P) // 1500)]
        s.pts = sub.astype(np.float32)
        s.centroid = P.mean(0); s.extent = np.ptp(P, 0)
        s.vox = vkeys(P, 0.2)
        s.desc = descriptor(sub)
        q_depth = float(np.exp(-np.median(z) / 6.0))
        q_pose = float(np.exp(-pose_sigma / 0.25))
        s.weight = s.score * d.get("mask_score", 1.0) * q_depth * q_pose
        s.cam = T_wc[:3, 3].copy()
        s.npix = int(d["mask"].sum())
        s.gt, s.gt_purity = 0, 0.0
        if gt_instance is not None:
            ids = gt_instance[vv, uu]
            vals, cnt = np.unique(ids, return_counts=True)
            j = int(np.argmax(cnt))
            s.gt, s.gt_purity = int(vals[j]), float(cnt[j] / cnt.sum())
            if s.gt_purity < 0.5:
                s.gt = 0
        segs.append(s)
    return segs


class Instance:
    def __init__(self, iid, seg, n_cls, uniform_weights=False):
        self.iid = iid
        self.uniform = uniform_weights
        self.scores = np.zeros(n_cls)
        self.vox = np.empty(0, np.int64)
        self.pts = np.empty((0, 3), np.float32)
        self.desc = None
        self.wsum = 0.0
        self.centroid = None
        self.views, self.segs = [], []
        self.hits, self.misses = 0, 0
        self.first, self.last = seg.frame, seg.frame
        self.add(seg)

    def add(self, s):
        w = 1.0 if self.uniform else max(s.weight, 1e-3)
        self.scores += w * s.probs
        self.centroid = s.centroid if self.centroid is None else (self.wsum * self.centroid + w * s.centroid) / (self.wsum + w)
        self.desc = s.desc if self.desc is None else (self.wsum * self.desc + w * s.desc) / (self.wsum + w)
        self.wsum += w
        self.vox = np.union1d(self.vox, s.vox)
        P = np.vstack([self.pts, s.pts])
        if len(P) > 6000:
            ijk = np.floor(P / 0.04).astype(np.int64)
            _, idx = np.unique(ijk, axis=0, return_index=True)
            P = P[idx]
        self.pts = P
        if all(np.linalg.norm(s.cam - v) > 0.25 for v in self.views):
            self.views.append(s.cam)
        self.segs.append(s)
        self.hits += 1
        self.last = s.frame

    @property
    def probs(self):
        return self.scores / max(self.scores.sum(), 1e-9)

    @property
    def extent(self):
        """Robust size (2nd-98th percentile) so a few stray points do not inflate the asset."""
        if len(self.pts) < 5:
            return np.ptp(self.pts, 0) if len(self.pts) else np.zeros(3)
        lo, hi = np.percentile(self.pts, [2, 98], axis=0)
        return hi - lo

    def robust_bounds(self, q=2):
        lo, hi = np.percentile(self.pts, [q, 100 - q], axis=0)
        return lo, hi

    def label_entropy(self):
        p = self.probs[self.probs > 0]
        return float(-(p * np.log(p)).sum())

    def existence(self):
        return self.hits / (self.hits + self.misses)


class GASF:
    def __init__(self, classes, w_sem=0.25, w_ov=0.35, w_cen=0.2, w_geo=0.2, accept=0.45,
                 confirm_obs=2, confirm_views=2, use_geometry=True, use_multiview=True, merge=True,
                 use_weights=True):
        self.classes = list(classes)
        self.w = np.array([w_sem, w_ov, w_cen, w_geo], float)
        if not use_geometry:
            self.w = np.array([0.5, 0.0, 0.5, 0.0])
        self.accept, self.confirm_obs, self.confirm_views = accept, confirm_obs, confirm_views
        self.use_geometry, self.use_multiview, self.do_merge = use_geometry, use_multiview, merge
        self.use_weights = use_weights          # False: every observation counts equally (ablation)
        self.instances = {}
        self.next_id = 1
        self.assign = {}            # segment id -> instance id (final, after merges)
        self.n_frames = 0

    def _score(self, s, inst):
        sem = float(s.probs @ inst.probs / (np.linalg.norm(inst.probs) + 1e-9))
        inter = np.intersect1d(s.vox, inst.vox, assume_unique=True).size
        ov = inter / max(min(s.vox.size, inst.vox.size), 1)
        size = max(s.extent.max(), inst.extent.max())
        sig = 0.25 + 0.25 * size
        d = np.linalg.norm(s.centroid - inst.centroid)
        cen = float(np.exp(-d * d / (2 * sig * sig)))
        geo = float(np.exp(-np.abs(s.desc - inst.desc).sum() / 2.0))
        S = float(self.w @ np.array([sem, ov, cen, geo]))
        if self.use_geometry:
            # strong 3D overlap with a similar shape overrides a (possibly wrong) 2D label
            ok = (sem >= 0.2 and S >= self.accept and (ov >= 0.1 or cen >= 0.5)) or (ov >= 0.5 and geo >= 0.5)
        else:
            ok = sem >= 0.2 and S >= self.accept
        return S, ok

    def update(self, segs, frame=None, K=None, T_wc=None):
        self.n_frames += 1
        active = [i for i in self.instances.values()]
        if segs and active:
            cost = np.full((len(segs), len(active)), 10.0)
            for a, s in enumerate(segs):
                for b, inst in enumerate(active):
                    if np.linalg.norm(s.centroid - inst.centroid) > 3.0 + s.extent.max() + inst.extent.max():
                        continue
                    S, ok = self._score(s, inst)
                    if ok:
                        cost[a, b] = 1.0 - S
            rows, cols = linear_sum_assignment(cost)
            matched = {r: c for r, c in zip(rows, cols) if cost[r, c] < 10.0}
        else:
            matched = {}
        touched = set()
        for a, s in enumerate(segs):
            if a in matched:
                inst = active[matched[a]]
                inst.add(s)
            else:
                inst = Instance(self.next_id, s, len(self.classes), uniform_weights=not self.use_weights)
                self.instances[self.next_id] = inst
                self.next_id += 1
            self.assign[s.sid] = inst.iid
            touched.add(inst.iid)
        if frame is not None and K is not None:
            self._visibility_misses(frame, K, T_wc, touched)
        if self.do_merge and self.n_frames % 5 == 0:
            self.merge_duplicates()

    def _visibility_misses(self, frame, K, T_wc, touched):
        """An instance that should be visible (in FOV, not occluded, close) but was not
        detected accumulates a miss; tentative ones with low existence are pruned."""
        h, w = frame["depth"].shape
        Tcw = np.linalg.inv(T_wc)
        for iid in list(self.instances):
            if iid in touched:
                continue
            inst = self.instances[iid]
            pc = Tcw[:3, :3] @ inst.centroid + Tcw[:3, 3]
            if pc[2] < 0.5 or pc[2] > 6.0:
                continue
            u = K[0, 0] * pc[0] / pc[2] + K[0, 2]; v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
            if not (0.1 * w < u < 0.9 * w and 0.1 * h < v < 0.9 * h):
                continue
            dz = frame["depth"][int(v), int(u)]
            if dz > 0 and dz < pc[2] - 0.4:                       # occluded
                continue
            inst.misses += 1
            if not self.is_confirmed(inst) and inst.misses >= 3 and inst.existence() < 0.4:
                del self.instances[iid]
                for s in inst.segs:
                    self.assign.pop(s.sid, None)

    def _should_merge(self, A, B):
        if np.linalg.norm(A.centroid - B.centroid) > 1.5 + max(A.extent.max(), B.extent.max()):
            return False
        inter = np.intersect1d(A.vox, B.vox, assume_unique=True).size
        iou = inter / max(np.union1d(A.vox, B.vox).size, 1)
        small = inter / max(min(A.vox.size, B.vox.size), 1)
        same = np.argmax(A.probs) == np.argmax(B.probs)
        if (same and (iou > 0.3 or small > 0.6)) or iou > 0.55:
            return True
        if same and self.use_geometry:
            # fragments of one object (split masks, partial views): they touch, and the union is
            # no larger than the bigger fragment's extent allows (rejects two adjacent objects)
            Bd = dilate_keys(B.vox) if B.vox.size < A.vox.size else B.vox
            Ad = A.vox if B.vox.size < A.vox.size else dilate_keys(A.vox)
            touch = np.intersect1d(Ad, Bd).size / max(min(A.vox.size, B.vox.size), 1)
            P = np.vstack([A.pts, B.pts])
            lo, hi = np.percentile(P, [2, 98], axis=0)
            union_ext = (hi - lo).max()
            if touch > 0.4 and union_ext <= 1.25 * max(A.extent.max(), B.extent.max()):
                return True
        return False

    def merge_duplicates(self):
        merged = True
        while merged:
            merged = False
            ids = sorted(self.instances)
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    A, B = self.instances[ids[a]], self.instances[ids[b]]
                    if self._should_merge(A, B):
                        keep, drop = (A, B) if A.wsum >= B.wsum else (B, A)
                        for s in drop.segs:
                            keep.add(s); self.assign[s.sid] = keep.iid
                        keep.hits -= len(drop.segs); keep.hits += drop.hits; keep.misses += drop.misses
                        del self.instances[drop.iid]
                        merged = True
                        break
                if merged:
                    break

    def is_confirmed(self, inst):
        if not self.use_multiview:
            return True
        return inst.hits >= self.confirm_obs and len(inst.views) >= self.confirm_views and inst.existence() >= 0.5

    def finalize(self):
        if self.do_merge:
            self.merge_duplicates()
        return [i for i in self.instances.values() if self.is_confirmed(i)]

    def objects(self):
        """Uniform output for evaluation: list of (label, [segments])."""
        return [(self.classes[int(np.argmax(i.probs))], i.segs, i) for i in self.finalize()]


# ----------------------------------------------------------------------------- baselines
def baseline_independent(all_segs, classes):
    """Independent 2D labelling: every detection becomes its own object."""
    return [(s.label, [s], None) for s in all_segs]


def baseline_centroid(all_segs, classes, radius=1.0, min_obs=2):
    """Greedy same-label nearest-centroid tracking (no 3D overlap / shape terms). Tracks seen fewer than
    min_obs times are dropped (same observation filter GASF uses, for a fair comparison)."""
    objs = []
    for s in sorted(all_segs, key=lambda x: x.frame):
        best, bd = None, radius
        for o in objs:
            if o["label"] != s.label:
                continue
            d = np.linalg.norm(o["c"] - s.centroid)
            if d < bd:
                best, bd = o, d
        if best is None:
            objs.append(dict(label=s.label, c=s.centroid.copy(), segs=[s]))
        else:
            n = len(best["segs"]); best["c"] = (best["c"] * n + s.centroid) / (n + 1); best["segs"].append(s)
    return [(o["label"], o["segs"], None) for o in objs if len(o["segs"]) >= min_obs]


def baseline_geometry_clustering(all_segs, classes, res=0.1, min_obs=2):
    """Geometry-only: connected components over all lifted points, majority label; clusters supported by
    fewer than min_obs detections are dropped."""
    if not all_segs:
        return []
    P = np.vstack([s.pts for s in all_segs])
    owner = np.concatenate([np.full(len(s.pts), k) for k, s in enumerate(all_segs)])
    ijk = np.floor(P / res).astype(np.int64)
    mn = ijk.min(0); ijk -= mn
    shape = ijk.max(0) + 1
    if np.prod(shape) > 5e7:
        res *= 2; ijk = np.floor(P / res).astype(np.int64); ijk -= ijk.min(0); shape = ijk.max(0) + 1
    grid = np.zeros(shape, bool); grid[tuple(ijk.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    pl = lab[tuple(ijk.T)]
    objs = []
    for c in range(1, n + 1):
        m = pl == c
        members = np.unique(owner[m])
        segs = [all_segs[k] for k in members]
        if len(segs) < min_obs:
            continue
        labels = [s.label for s in segs]
        lab_major = max(set(labels), key=labels.count)
        objs.append((lab_major, segs, None))
    return objs
