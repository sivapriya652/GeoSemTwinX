"""Synthetic-warehouse benchmark (CPU, exact ground truth): SLAM, GASF vs baselines, change detection, NBV."""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

from .change import detect_changes, score_changes
from .data import SyntheticSequence
from .detectors import OracleNoisyDetector
from .evaluation import evaluate_objects, gt_voxel_labels, object_gt, semantic_miou
from .gasf import (GASF, baseline_centroid, baseline_geometry_clustering, baseline_independent)
from .metrics import ate, path_length, rpe
from .pipeline import run_mapping, visible_gt_assets
from .slam import run_slam
from .synthetic import ASSET_CLASSES

OUT = Path("results")


def set_out(path):
    global OUT
    OUT = Path(path); OUT.mkdir(parents=True, exist_ok=True)
BOUNDS = ((-0.5, -0.5, -0.2), (24.5, 16.5, 6.0))


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))


def exp_slam(n=720, scale=1.0):
    seq = SyntheticSequence("A", n, stride=2, scale=scale)
    n = len(seq)
    T0 = np.linalg.inv(seq.gt_poses[0]); gt = [T0 @ T for T in seq.gt_poses]
    t = time.time()
    slam, _ = run_slam(seq, depth_max=8.0)
    el = time.time() - t
    res = dict(frames=n, resolution=list(seq.frame(0)["depth"].shape), path_m=path_length(gt),
               keyframes=len(slam.keyframes), loops=slam.n_loops, track_fallbacks=slam.n_track_fail,
               time_per_frame_ms=1000 * el / n, opt_time_s=slam.t_opt)
    a = ate(slam.poses, gt); r = rpe(slam.poses, gt, seq.timestamps)
    a0 = ate(slam.poses_before_opt, gt); r0 = rpe(slam.poses_before_opt, gt, seq.timestamps)
    res.update(ate_rmse=a["rmse"], ate_median=a["median"], ate_max=a["max"], rpe_trans=r["trans_rmse"],
               rpe_rot_deg=r["rot_rmse_deg"], odo_ate_rmse=a0["rmse"], odo_rpe_trans=r0["trans_rmse"],
               drift_pct=100 * a["rmse"] / path_length(gt))
    save("slam", res)
    np.savez(OUT / "slam_traj.npz", est=np.array([T for T in slam.poses]), odo=np.array(slam.poses_before_opt),
             gt=np.array(gt), sig=np.array(slam.sigmas), sig_step=np.array(slam.step_sigmas), T0=np.linalg.inv(T0))
    print(json.dumps(res, indent=1))


def _semantic_run(seq, poses, sigmas, seed, gasf_kwargs=None, stride=3):
    det = OracleNoisyDetector(ASSET_CLASSES, seq.instance_class, seed=seed)
    return run_mapping(seq, poses, sigmas, det, ASSET_CLASSES, gasf_stride=stride, occ_bounds=BOUNDS,
                       gasf_kwargs=gasf_kwargs, verbose=False)


def exp_semantic(n=720, scale=0.5, seeds=(0, 1, 2), use_slam=True):
    seq = SyntheticSequence("A", n, stride=2, scale=scale)
    n = len(seq)
    poses, sigmas, T_align = seq.gt_poses, None, np.eye(4)
    if use_slam and (OUT / "slam_traj.npz").exists():
        d = np.load(OUT / "slam_traj.npz")
        T0inv = d["T0"]
        poses = [T0inv @ T for T in d["est"]][:n]            # SLAM poses in GT world frame (anchored at frame 0)
        sigmas = d["sig_step"][:n] if "sig_step" in d else None
    frames = list(range(0, n, 3))
    gt_vox = gt_voxel_labels(seq, frames, seq.instance_class, ASSET_CLASSES)
    methods = {"GASF (ours)": {}, "GASF w/o geometry terms": dict(use_geometry=False),
               "GASF w/o multi-view gate": dict(use_multiview=False), "GASF w/o duplicate merge": dict(merge=False)}
    rows = []
    for seed in seeds:
        out = None
        for name, kw in methods.items():
            o = _semantic_run(seq, poses, sigmas, seed, kw)
            out = out or o
            vis = visible_gt_assets(seq, o["gasf_frames"])
            m = evaluate_objects(o["gasf"].objects(), o["segments"], seq.instance_class, vis)
            m.update(semantic_miou(o["gasf"].objects(), gt_vox, ASSET_CLASSES))
            m.update(method=name, seed=seed, timings=o["timings_ms"])
            rows.append(m); print(name, seed, round(m["f1"], 3), round(m["identity_consistency"], 3), round(m["miou"], 3))
        segs = out["segments"]; vis = visible_gt_assets(seq, out["gasf_frames"])
        for name, fn in (("Independent 2D labelling", baseline_independent), ("Centroid tracking", baseline_centroid),
                         ("Geometry-only clustering", baseline_geometry_clustering)):
            objs = fn(segs, ASSET_CLASSES)
            m = evaluate_objects(objs, segs, seq.instance_class, vis)
            m.update(semantic_miou(objs, gt_vox, ASSET_CLASSES))
            m.update(method=name, seed=seed); rows.append(m)
            print(name, seed, round(m["f1"], 3), round(m["identity_consistency"], 3), round(m["miou"], 3))
    save("semantic_slam" if use_slam else "semantic_gtpose", rows)


def _assets(gasf, seq):
    out, names = [], {}
    for k, (label, segs, inst) in enumerate(gasf.objects()):
        g, _ = object_gt(segs)
        aid = f"A{k:03d}"
        out.append(dict(id=aid, label=label, centroid=inst.pts.mean(0), extent=np.ptp(inst.pts, 0), pts=inst.pts))
        names[aid] = seq.instance_names[g] if g else None
    return out, names


def exp_change(n=720, scale=0.5, seeds=(0, 1, 2)):
    rows = []
    seqA = SyntheticSequence("A", n, stride=2, scale=scale); seqB = SyntheticSequence("B", n, stride=2, scale=scale)
    for seed in seeds:
        oA = _semantic_run(seqA, seqA.gt_poses, None, seed)
        oB = _semantic_run(seqB, seqB.gt_poses, None, seed + 100)
        aA, nA = _assets(oA["gasf"], seqA); aB, nB = _assets(oB["gasf"], seqB)
        pred = detect_changes(aA, aB, oA["occ"], oB["occ"])
        sc = score_changes(pred, seqB.changes, nA, nB)
        sc_no_vis = score_changes(detect_changes(aA, aB, None, None), seqB.changes, nA, nB)
        rows.append(dict(seed=seed, **{k: v for k, v in sc.items()},
                         no_visibility=dict(precision=sc_no_vis["precision"], recall=sc_no_vis["recall"],
                                            f1=sc_no_vis["f1"]),
                         raw=[{k: v for k, v in p.items()} for p in pred]))
        print(seed, sc["precision"], sc["recall"], sc["pred"], "| no-vis", sc_no_vis["precision"], sc_no_vis["recall"])
    save("change", rows)


def exp_nbv(budget=80.0, seeds=(0, 1, 2),
            strategies=("nbv", "nbv_no_semantic", "nbv_coverage_only", "frontier", "random")):
    from .nbv import Explorer
    from .synthetic import SyntheticWarehouse, default_layout
    rows = {}
    starts = [(3.5, 8.0, 0.0), (20.0, 8.0, np.pi), (12.0, 2.2, 0.0)]
    for seed in seeds:
        for st in strategies:
            objs = default_layout(0)
            sim = SyntheticWarehouse(objs, width=320, height=240, fx=262.5, fy=262.5, cx=159.5, cy=119.5, seed=seed)
            ic = {sim.inst_id[o["name"]]: o["cls"] for o in objs if o["cls"] in ASSET_CLASSES}
            g = GASF(ASSET_CLASSES); det = OracleNoisyDetector(ASSET_CLASSES, ic, seed=seed)
            ex = Explorer(sim, st, seed=seed, gasf=g, detector=det, classes=ASSET_CLASSES)
            t = time.time(); log = ex.run(start=starts[seed % 3], budget_m=budget)
            confirmed = [i for i in g.instances.values() if g.is_confirmed(i)]
            ident = set()
            for inst in confirmed:                       # assets identified with the correct class
                gt, pur = object_gt(inst.segs)
                if gt and pur >= 0.5 and ic.get(gt) == ASSET_CLASSES[int(np.argmax(inst.probs))]:
                    ident.add(gt)
            rows[f"{st}_{seed}"] = dict(strategy=st, seed=seed, log=log, total_assets=len(ic),
                                        confirmed_instances=len(confirmed), assets_identified=len(ident),
                                        time_s=time.time() - t)
            print(st, seed, "cov", round(log[-1]["coverage"], 3), "assets", log[-1]["assets_seen"], "/", len(ic),
                  "replans", log[-1]["replans"], "coll", log[-1]["collisions"], round(time.time() - t, 1), "s")
            save("nbv", rows)


