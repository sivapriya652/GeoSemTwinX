"""Comparative studies.

    cache_detections   run the (expensive) 2D detector once per keyframe and reuse it everywhere
    fusion_study       GASF vs conventional association methods + GASF ablations
    pose_noise_study   robustness of every method to pose error (what SLAM drift does to the twin)
    threshold_study    sensitivity to the detector confidence threshold
    slam_study         SLAM configurations / ablations on any sequences, next to published numbers
    plots + tables     PNG charts and markdown tables for the report / dashboard
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate_objects, gt_voxel_labels, semantic_miou
from .gasf import GASF, baseline_centroid, baseline_geometry_clustering, baseline_independent, make_segments
from .metrics import ate, path_length, rpe
from .pipeline import visible_gt_assets
from .slam import run_slam

# ATE RMSE (m) as reported in ORB-SLAM2 (Mur-Artal & Tardos, T-RO 2017, Table III) for the RGB-D methods.
PUBLISHED_ATE = {
    "rgbd_dataset_freiburg1_desk": {"ORB-SLAM2": 0.016, "ElasticFusion": 0.020, "DVO-SLAM": 0.021},
    "rgbd_dataset_freiburg2_xyz": {"ORB-SLAM2": 0.004, "ElasticFusion": 0.011, "DVO-SLAM": 0.018},
    "rgbd_dataset_freiburg3_long_office_household": {"ORB-SLAM2": 0.010, "ElasticFusion": 0.017, "DVO-SLAM": 0.035},
}

METHOD_ORDER = ["GASF (ours)", "GASF w/o geometry terms", "GASF w/o multi-view gate", "GASF w/o duplicate merge",
                "GASF w/o confidence weighting", "Centroid tracking", "Geometry-only clustering", "Independent 2D labelling"]
PALETTE = {"GASF (ours)": "#1F5BD8", "Centroid tracking": "#D9731A", "Geometry-only clustering": "#2A9D7F",
           "Independent 2D labelling": "#A04FB0", "GASF w/o geometry terms": "#7FA3EA",
           "GASF w/o multi-view gate": "#9DB8EE", "GASF w/o duplicate merge": "#BACDF3",
           "GASF w/o confidence weighting": "#D2DEF7"}
GASF_VARIANTS = {"GASF (ours)": {}, "GASF w/o geometry terms": dict(use_geometry=False),
                 "GASF w/o multi-view gate": dict(use_multiview=False), "GASF w/o duplicate merge": dict(merge=False),
                 "GASF w/o confidence weighting": dict(use_weights=False)}
BASELINES = {"Centroid tracking": baseline_centroid, "Geometry-only clustering": baseline_geometry_clustering,
             "Independent 2D labelling": baseline_independent}
KEYS = ["objects", "precision", "recall", "f1", "duplicate_rate", "identity_consistency", "id_switches",
        "label_acc", "mean_purity", "miou"]


# ---------------------------------------------------------------- detections
def cache_detections(seq, detector, frames, on="rgb", verbose=True):
    cache, t = {}, []
    for n, i in enumerate(frames):
        f = seq.frame(i)
        a = time.perf_counter()
        cache[i] = detector(f["rgb"] if on == "rgb" else f)
        t.append(time.perf_counter() - a)
        if verbose and n % 25 == 0:
            print(f"  detections {n}/{len(frames)}")
    cache_ms = 1000 * float(np.mean(t)) if t else 0.0
    return cache, cache_ms


def save_cache(path, cache):
    out = {}
    for i, dets in cache.items():
        for k, d in enumerate(dets):
            out[f"{i}_{k}_mask"] = np.packbits(d["mask"])
            out[f"{i}_{k}_meta"] = np.array([d["score"], d.get("mask_score", 1.0)] + list(d["mask"].shape), float)
            out[f"{i}_{k}_label"] = np.array(d["label"])
    np.savez_compressed(path, **out)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    cache = {}
    metas = sorted((tuple(map(int, key.split("_")[:2])), key) for key in z.files if key.endswith("_meta"))
    for (i, k), key in metas:
        meta = z[key]; h, w = int(meta[2]), int(meta[3])
        mask = np.unpackbits(z[f"{i}_{k}_mask"])[: h * w].reshape(h, w).astype(bool)
        cache.setdefault(i, []).append(dict(label=str(z[f"{i}_{k}_label"]), score=float(meta[0]),
                                            mask_score=float(meta[1]), mask=mask))
    return cache


# ---------------------------------------------------------------- fusion
def lift(seq, cache, frames, poses, classes, sigmas=None, seg_kwargs=None, max_depth=8.0, use_gt=True):
    by_frame, n = {}, 0
    for i in frames:
        f = seq.frame(i)
        segs = make_segments(f, cache.get(i, []), seq.K, poses[i], classes, i,
                             pose_sigma=sigmas[i] if sigmas is not None else 0.0, max_depth=max_depth,
                             gt_instance=f.get("instance") if use_gt else None, start_sid=n, **(seg_kwargs or {}))
        by_frame[i] = segs; n += len(segs)
    return by_frame


def fuse(seq, by_frame, frames, poses, classes, **kw):
    g = GASF(classes, **kw)
    for i in frames:
        g.update(by_frame[i], seq.frame(i), seq.K, poses[i])
    return g


def evaluate_all(seq, by_frame, frames, poses, classes, visible, gt_vox=None, methods=None):
    segs = [s for i in frames for s in by_frame[i]]
    rows = {}
    for name in (methods or METHOD_ORDER):
        if name in GASF_VARIANTS:
            objs = fuse(seq, by_frame, frames, poses, classes, **GASF_VARIANTS[name]).objects()
        else:
            objs = BASELINES[name](segs, classes)
        m = evaluate_objects(objs, segs, seq.instance_class, visible)
        if gt_vox is not None:
            m.update(semantic_miou(objs, gt_vox, classes))
        rows[name] = m
    return rows


def _rows_to_df(rows, **extra):
    recs = []
    for name, m in rows.items():
        r = {"method": name, **extra}
        r.update({k: m.get(k, np.nan) for k in KEYS})
        recs.append(r)
    return pd.DataFrame(recs)


def fusion_study(seq, cache, frames, classes, pose_sets, seg_kwargs=None, max_depth=8.0, visible_kw=None):
    """pose_sets: {name: (poses, sigmas)} e.g. {'ground-truth poses': (gt, None), 'SLAM poses': (est, sig)}"""
    visible = visible_gt_assets(seq, frames, max_depth=max_depth, **(visible_kw or {}))
    gt_vox = gt_voxel_labels(seq, frames, seq.instance_class, classes, max_depth=max_depth, include_ids=visible)
    dfs = []
    for pname, (P, S) in pose_sets.items():
        by_frame = lift(seq, cache, frames, P, classes, S, seg_kwargs, max_depth)
        dfs.append(_rows_to_df(evaluate_all(seq, by_frame, frames, P, classes, visible, gt_vox), poses=pname))
    df = pd.concat(dfs, ignore_index=True)
    df.attrs["visible_assets"] = len(visible)
    return df, visible


def perturb_poses(poses, sigma_t, seed=0, drift=True):
    """Pose error model. drift=True: random-walk drift (how SLAM error behaves); False: per-frame jitter.
    sigma_t is the RMS translation error reached at the end of the sequence (metres); rotation
    error scales as 2 degrees per 0.1 m."""
    rng = np.random.default_rng(seed)
    n = len(poses)
    from scipy.spatial.transform import Rotation
    if drift:
        steps_t = rng.normal(0, sigma_t / np.sqrt(n), (n, 3)); steps_r = rng.normal(0, np.radians(20 * sigma_t) / np.sqrt(n), (n, 3))
        dt, dr = np.cumsum(steps_t, 0), np.cumsum(steps_r, 0)
    else:
        dt = rng.normal(0, sigma_t, (n, 3)); dr = rng.normal(0, np.radians(20 * sigma_t), (n, 3))
    out = []
    for T, t, r in zip(poses, dt, dr):
        E = np.eye(4); E[:3, :3] = Rotation.from_rotvec(r).as_matrix(); E[:3, 3] = t
        out.append(T @ E)
    return out


def pose_noise_study(seq, cache, frames, classes, levels=(0.0, 0.05, 0.1, 0.2, 0.4), seeds=(0, 1),
                     methods=("GASF (ours)", "Centroid tracking", "Geometry-only clustering", "Independent 2D labelling"),
                     seg_kwargs=None, max_depth=8.0, drift=True):
    visible = visible_gt_assets(seq, frames, max_depth=max_depth)
    dfs = []
    for lv in levels:
        for sd in (seeds if lv > 0 else (0,)):
            P = perturb_poses(seq.gt_poses, lv, sd, drift) if lv > 0 else seq.gt_poses
            by_frame = lift(seq, cache, frames, P, classes, None, seg_kwargs, max_depth)
            dfs.append(_rows_to_df(evaluate_all(seq, by_frame, frames, P, classes, visible, None, list(methods)),
                                   pose_error_m=lv, seed=sd))
    df = pd.concat(dfs, ignore_index=True)
    return df.groupby(["pose_error_m", "method"], as_index=False).mean(numeric_only=True)


def threshold_study(seq, cache, frames, classes, thresholds=(0.25, 0.30, 0.35, 0.40, 0.45),
                    methods=("GASF (ours)", "Centroid tracking", "Independent 2D labelling"), seg_kwargs=None,
                    max_depth=8.0):
    visible = visible_gt_assets(seq, frames, max_depth=max_depth)
    dfs = []
    for th in thresholds:
        kw = dict(seg_kwargs or {}, min_score=th)
        by_frame = lift(seq, cache, frames, seq.gt_poses, classes, None, kw, max_depth)
        dfs.append(_rows_to_df(evaluate_all(seq, by_frame, frames, seq.gt_poses, classes, visible, None,
                                            list(methods)), threshold=th))
    return pd.concat(dfs, ignore_index=True)


def keyframe_interval_study(seq, cache_all, classes, intervals=(1, 2, 3, 5, 10), poses=None, seg_kwargs=None,
                            max_depth=8.0, det_ms=None, other_ms=None,
                            methods=("GASF (ours)", "Centroid tracking", "Independent 2D labelling")):
    """How often must the (expensive) detector run? cache_all holds detections for every frame.
    Reports accuracy per interval plus the detector cost per input frame (det_ms / interval)."""
    poses = poses or seq.gt_poses
    dfs = []
    for k in intervals:
        frames = [i for i in range(0, len(seq), k) if i in cache_all]
        visible = visible_gt_assets(seq, frames, max_depth=max_depth)
        by_frame = lift(seq, cache_all, frames, poses, classes, None, seg_kwargs, max_depth)
        df = _rows_to_df(evaluate_all(seq, by_frame, frames, poses, classes, visible, None, list(methods)),
                         interval=k, keyframes=len(frames))
        if det_ms is not None:
            df["detector_ms_per_frame"] = det_ms / k
            df["pipeline_ms_per_frame"] = det_ms / k + (other_ms or 0.0)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def map_quality_study(seq, pose_sets, frames=None, configs=None, voxel=0.05, gt_stride=4, tol=0.05, max_depth=10.0):
    """Reconstruction accuracy / completeness of the fused map against the ground-truth surface.
    pose_sets: {name: (poses, step_sigmas)}. configs: {name: FusedMap kwargs}. Default configs compare our
    uncertainty-aware fusion (noise-based depth cut + pose gating + weighting) with naive averaging."""
    from scipy.spatial import cKDTree
    from .mapping import FusedMap, backproject
    frames = frames or list(range(len(seq)))
    configs = configs or {
        "Uncertainty-aware fusion (ours)": dict(sigma_max=0.08, pose_sigma_gate=0.15, uncertainty_weighting=True),
        "Naive fusion (no gating, no weights)": dict(sigma_max=10.0, pose_sigma_gate=1e9, uncertainty_weighting=False)}
    G = [backproject(seq.frame(i)["depth"], seq.K, seq.gt_poses[i], stride=gt_stride, max_depth=max_depth)[0] for i in frames]
    G = np.vstack(G)
    if len(G) > 3_000_000:
        G = G[np.random.default_rng(0).choice(len(G), 3_000_000, replace=False)]
    gtree = cKDTree(G)
    h, w = seq.frame(frames[0])["depth"].shape
    rows = []
    for pname, (P, S) in pose_sets.items():
        for cname, kw in configs.items():
            fm = FusedMap(seq.K, w, h, voxel=voxel, depth_max=max_depth, **kw)
            for i in frames:
                fm.integrate(seq.frame(i), P[i], S[i] if S is not None else 0.0)
            M = fm.arrays()[0]
            if len(M) == 0:
                continue
            d, _ = gtree.query(M)
            mtree = cKDTree(M)
            Gs = G[np.random.default_rng(1).choice(len(G), min(400000, len(G)), replace=False)]
            dc, _ = mtree.query(Gs)
            rows.append(dict(poses=pname, method=cname, map_points=len(M), accuracy_median_m=float(np.median(d)),
                             accuracy_p90_m=float(np.percentile(d, 90)), precision_at_tol=float((d < tol).mean()),
                             completeness_at_tol=float((dc < tol).mean()), frames_skipped=fm.n_skipped))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- SLAM
SLAM_CONFIGS = {
    "Full (hybrid + PnP + loops)": dict(tracker="hybrid"),
    "w/o PnP recovery": dict(tracker="hybrid", use_pnp=False),
    "ICP tracker (geometry only)": dict(tracker="icp"),
}
# derived for free from the runs above (pose graph not optimised)
DERIVED = {"w/o loop closure": "Full (hybrid + PnP + loops)", "Odometry only (no PnP, no loops)": "w/o PnP recovery"}


def _slam_row(sname, cname, P, seq, extra):
    a, r = ate(P, seq.gt_poses), rpe(P, seq.gt_poses, seq.timestamps)
    return dict(sequence=sname, method=cname, ate_m=a["rmse"], rpe_m=r["trans_rmse"], rpe_deg=r["rot_rmse_deg"],
                path_m=path_length(seq.gt_poses), **extra)


def slam_study(sequences, configs=None, depth_max=4.0, verbose=True, derived=True, keep_poses=False):
    """sequences: {name: seq}. Returns (DataFrame with ATE/RPE per sequence x configuration, poses dict)."""
    rows, poses = [], {}
    for sname, seq in sequences.items():
        runs, ms = {}, {}
        for cname, kw in (configs or SLAM_CONFIGS).items():
            t = time.time()
            slam, P = run_slam(seq, depth_max=kw.get("depth_max", depth_max), verbose=False,
                               **{k: v for k, v in kw.items() if k != "depth_max"})
            el = time.time() - t
            runs[cname] = slam; ms[cname] = 1000 * el / len(seq)
            poses[(sname, cname)] = (P, slam.step_sigmas)
            rows.append(_slam_row(sname, cname, P, seq, dict(loops=slam.n_loops, fallbacks=slam.n_track_fail,
                                                              ms_per_frame=1000 * el / len(seq))))
            if verbose:
                print(f"  {sname:45s} {cname:32s} ATE {100 * rows[-1]['ate_m']:.1f} cm  ({el / 60:.1f} min)")
        if derived:
            for dname, src in DERIVED.items():
                if src in runs:
                    rows.append(_slam_row(sname, dname, runs[src].poses_before_opt, seq,
                                          dict(loops=0, fallbacks=runs[src].n_track_fail,
                                               ms_per_frame=ms[src])))
        for m, v in PUBLISHED_ATE.get(sname, {}).items():
            rows.append(dict(sequence=sname, method=f"{m} (published)", ate_m=v))
    df = pd.DataFrame(rows)
    return (df, poses) if keep_poses else df


# ---------------------------------------------------------------- plots & tables
def _style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, color="#1d2433")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9ced8"); ax.spines["bottom"].set_color("#c9ced8")
    ax.tick_params(colors="#5b6577"); ax.grid(axis="y", color="#e6e9ef", lw=0.8); ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color="#5b6577")


def plot_methods(df, path, metrics=("precision", "recall", "f1", "identity_consistency", "miou"),
                 methods=("GASF (ours)", "Centroid tracking", "Geometry-only clustering", "Independent 2D labelling"),
                 title="Semantic fusion vs conventional methods"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.groupby("method").mean(numeric_only=True)
    methods = [m for m in methods if m in d.index]
    metrics = [m for m in metrics if m in d.columns and d[m].notna().any()]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.8 / len(methods); x = np.arange(len(metrics))
    for k, m in enumerate(methods):
        v = d.loc[m, list(metrics)].values.astype(float)
        bars = ax.bar(x + k * w - 0.4 + w / 2, v, w * 0.92, color=PALETTE.get(m, "#888"), label=m)
        if m == "GASF (ours)":
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", fontsize=8,
                        color="#1d2433", weight="bold")
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics]); ax.set_ylim(0, 1.08)
    _style(ax, title); ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_ablation(df, path, metric="f1", title="GASF ablation (F1)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.groupby("method").mean(numeric_only=True)
    order = [m for m in METHOD_ORDER if m in d.index]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    vals = [d.loc[m, metric] for m in order]
    ax.barh(order[::-1], vals[::-1], color=[PALETTE.get(m, "#888") for m in order[::-1]])
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9, color="#1d2433")
    ax.set_xlim(0, 1.05); _style(ax, title); ax.grid(axis="x", color="#e6e9ef"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_lines(df, x, y, path, title, xlabel, ylabel, methods=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ms = methods or [m for m in METHOD_ORDER if m in set(df["method"])]
    for m in sorted(ms, key=lambda m: m == "GASF (ours)"):            # draw ours last so it stays visible
        d = df[df["method"] == m].sort_values(x)
        if len(d):
            ours = m == "GASF (ours)"
            ax.plot(d[x], d[y], marker="o", ms=6 if ours else 4, lw=2.6 if ours else 1.6, zorder=3 if ours else 2,
                    color=PALETTE.get(m, "#888"), label=m, ls="-" if ours else "--")
    ax.set_xlabel(xlabel, color="#5b6577"); _style(ax, title, ylabel); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_slam(df, path, title="Trajectory error (ATE RMSE)"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seqs = list(dict.fromkeys(df["sequence"]))
    fig, axes = plt.subplots(1, len(seqs), figsize=(4.6 * len(seqs), 3.8), squeeze=False)
    for ax, s in zip(axes[0], seqs):
        d = df[df["sequence"] == s].dropna(subset=["ate_m"])
        cols = ["#9aa3b2" if "published" in m else ("#1F5BD8" if m.startswith("Full") else "#7FA3EA") for m in d["method"]]
        ax.barh(d["method"][::-1], 100 * d["ate_m"][::-1], color=cols[::-1])
        for yy, v in enumerate(100 * d["ate_m"][::-1]):
            ax.text(v, yy, f" {v:.1f}", va="center", fontsize=8)
        _style(ax, s.replace("rgbd_dataset_", "")); ax.set_xlabel("ATE (cm)", color="#5b6577")
        ax.grid(axis="x", color="#e6e9ef"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def table_md(df, cols=None, floatfmt=3):
    d = df[cols] if cols else df
    try:
        return d.to_markdown(index=False, floatfmt=f".{floatfmt}f")
    except ImportError:
        return d.round(floatfmt).to_string(index=False)


def save_study(outdir, name, df):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{name}.csv", index=False)
    (outdir / f"{name}.json").write_text(df.to_json(orient="records", indent=1))
    return outdir / f"{name}.csv"
