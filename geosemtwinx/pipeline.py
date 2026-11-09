"""End-to-end GeoSemTwin-X run: poses -> TSDF + occupancy -> GASF -> digital twin."""
import time

import numpy as np

from .gasf import GASF, make_segments
from .mapping import OccupancyMap, TSDFMap


def peak_ram_gb():
    """Peak resident memory of this process in GB (Linux/WSL via resource, Windows via psutil)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    except ImportError:
        try:
            import psutil
            mi = psutil.Process().memory_info()
            return getattr(mi, "peak_wset", mi.rss) / 1024 ** 3
        except ImportError:
            return float("nan")


def run_mapping(seq, poses, sigmas, detector, classes, gasf_stride=5, tsdf_voxel=0.02, tsdf_depth=4.0,
                occ_bounds=None, occ_res=0.1, max_depth=8.0, gasf_kwargs=None, use_gt_eval=True,
                verbose=True, detector_on="frame", seg_kwargs=None, det_cache=None, map_sigma_max=0.03):
    """detector: callable (frame or rgb) -> detections, or None when det_cache {frame index: dets} is given."""
    f0 = seq.frame(0)
    h, w = f0["depth"].shape
    tsdf = TSDFMap(seq.K, w, h, voxel=tsdf_voxel, depth_max=tsdf_depth, sigma_max=map_sigma_max)
    if occ_bounds is None:
        C = np.array([T[:3, 3] for T in poses])
        occ_bounds = (C.min(0) - max_depth, C.max(0) + max_depth)
    occ = OccupancyMap(occ_bounds[0], occ_bounds[1], occ_res, max_range=max_depth)
    gasf = GASF(classes, **(gasf_kwargs or {}))
    segs_all, t = [], dict(tsdf=[], occ=[], detect=[], lift=[], fuse=[])
    gasf_frames = []
    for i in range(len(seq)):
        f = seq.frame(i)
        T, sg = poses[i], (sigmas[i] if sigmas is not None else 0.0)
        a = time.perf_counter(); tsdf.integrate(f, T, sg); t["tsdf"].append(time.perf_counter() - a)
        if i % gasf_stride:
            continue
        gasf_frames.append(i)
        a = time.perf_counter(); occ.integrate(f["depth"], seq.K, T, stride=8, pose_sigma=sg)
        t["occ"].append(time.perf_counter() - a)
        a = time.perf_counter()
        if det_cache is not None:
            dets = det_cache.get(i, [])
        else:
            dets = detector(f if detector_on == "frame" else f["rgb"])
        t["detect"].append(time.perf_counter() - a)
        a = time.perf_counter()
        segs = make_segments(f, dets, seq.K, T, classes, i, pose_sigma=sg, max_depth=max_depth,
                             gt_instance=f.get("instance") if use_gt_eval else None, start_sid=len(segs_all),
                             **(seg_kwargs or {}))
        t["lift"].append(time.perf_counter() - a)
        a = time.perf_counter(); gasf.update(segs, f, seq.K, T); t["fuse"].append(time.perf_counter() - a)
        segs_all += segs
        if verbose and len(gasf_frames) % 20 == 0:
            print(f"  frame {i}/{len(seq)}  segments={len(segs_all)}  instances={len(gasf.instances)}")
    timings = {k: float(np.mean(v)) * 1000 if v else 0.0 for k, v in t.items()}
    return dict(tsdf=tsdf, occ=occ, gasf=gasf, segments=segs_all, gasf_frames=gasf_frames,
                timings_ms=timings, peak_ram_gb=peak_ram_gb())


def visible_gt_assets(seq, frame_ids, min_pixels=None, min_frames=2, max_depth=8.0, min_frac=0.002):
    """GT assets a detector could reasonably find: in >= min_frames keyframes, each time covering
    >= min_pixels (default 0.2% of the image) with median depth <= max_depth."""
    cnt = {}
    for i in frame_ids:
        f = seq.frame(i)
        inst, depth = f["instance"], f["depth"]
        mp = min_pixels or int(min_frac * inst.size)
        ids, c = np.unique(inst, return_counts=True)
        for a, b in zip(ids, c):
            if not a or b < mp or a not in (seq.instance_class or {}):
                continue
            z = depth[inst == a]
            z = z[z > 0]
            if len(z) and np.median(z) <= max_depth:
                cnt[a] = cnt.get(a, 0) + 1
    return sorted(int(a) for a, n in cnt.items() if n >= min_frames)
