"""All experiments for the GeoSemTwin-X prototype, as functions called by scripts/run_all.py.

Every experiment writes CSV/JSON tables to <out>/comparative, PNG charts to <out>/figures and appends cards to
<out>/studies.json (picked up by the dashboard and the final report).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import compare as CMP
from .dashboard import build_dashboard, study
from .metrics import ate, path_length, rpe, umeyama
from .pipeline import run_mapping, visible_gt_assets
from .power import GPUMonitor, idle_power
from .slam import run_slam
from .twin import DigitalTwin, TwinStore

STAER_CLASSES = ["forklift", "pallet", "cardboard box", "pallet truck"]
FLOOR_CLASSES = ["forklift", "pallet truck"]
TUM_CLASSES = ["monitor", "keyboard", "computer mouse", "cup", "book", "chair", "telephone", "bottle"]


class Out:
    def __init__(self, root):
        self.root = Path(root); self.cmp = self.root / "comparative"; self.fig = self.root / "figures"
        for d in (self.root, self.cmp, self.fig):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, name, df):
        return CMP.save_study(self.cmp, name, df)

    def add_study(self, key, title, note="", df=None, image=None, highlight="GASF (ours)"):
        p = self.root / "studies.json"
        cards = json.loads(p.read_text()) if p.exists() else {}
        cards[key] = dict(title=title, note=note, csv=None, image=str(image) if image else None, highlight=highlight)
        if df is not None:
            csv = self.cmp / f"{key}_card.csv"; df.to_csv(csv, index=False); cards[key]["csv"] = str(csv)
        p.write_text(json.dumps(cards, indent=1))

    def studies(self):
        p = self.root / "studies.json"
        out = []
        for c in (json.loads(p.read_text()) if p.exists() else {}).values():
            df = pd.read_csv(c["csv"]) if c.get("csv") and Path(c["csv"]).exists() else None
            out.append(study(c["title"], c["note"], df, c["image"] if c.get("image") and Path(c["image"]).exists() else None,
                             c.get("highlight", "GASF (ours)")))
        return out

    def merge_json(self, name, obj):
        p = self.root / f"{name}.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        d.update(obj); p.write_text(json.dumps(d, indent=1, default=float))


def make_detector(classes, dino="tiny", sam="tiny", threshold=0.20, ckpt_dir="checkpoints"):
    from .detectors import GroundedSAM2Detector
    dino_id = {"tiny": "IDEA-Research/grounding-dino-tiny", "base": "IDEA-Research/grounding-dino-base"}.get(dino, dino)
    return GroundedSAM2Detector(classes, dino_id=dino_id, sam_size=sam, box_threshold=threshold, ckpt_dir=ckpt_dir)


def plot_trajectories(path, gt, trajs, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5))
    G = np.array([T[:3, 3] for T in gt]); ax.plot(G[:, 0], G[:, 1], color="#1d2433", lw=2.2, label="ground truth")
    cols = ["#1F5BD8", "#D9731A", "#2A9D7F", "#A04FB0"]
    for (name, P), c in zip(trajs.items(), cols):
        a = ate(P, gt); A = a["aligned"]
        ax.plot(A[:, 0], A[:, 1], color=c, lw=1.6, label=f"{name}  (ATE {100 * a['rmse']:.1f} cm)")
    ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=9); CMP._style(ax, title)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


# =============================================================================== TUM
def run_tum(out, seqs=("rgbd_dataset_freiburg1_desk",), configs=("Full (hybrid + PnP + loops)", "w/o PnP recovery"),
            root="data/tum", detector_args=None, semantic=True):
    from .data import TUMSequence
    O = Out(out)
    tums = {s: TUMSequence(s, root=root) for s in seqs}
    df, poses = CMP.slam_study(tums, {k: CMP.SLAM_CONFIGS[k] for k in configs}, depth_max=4.0, keep_poses=True)
    O.save("tum_slam", df); CMP.plot_slam(df, O.fig / "tum_slam.png")
    O.add_study("tum_slam", "SLAM accuracy on TUM RGB-D (ATE RMSE)",
                "full system + component ablations vs published RGB-D SLAM (grey)",
                df[["sequence", "method", "ate_m", "rpe_m", "loops", "ms_per_frame"]], O.fig / "tum_slam.png",
                highlight="Full (hybrid + PnP + loops)")
    s0 = seqs[0]; tum = tums[s0]; P, S = poses[(s0, configs[0])]
    plot_trajectories(O.fig / "tum_trajectory.png", tum.gt_poses, {"GeoSemTwin-X": P}, f"{s0}: top view")
    print(CMP.table_md(df[["sequence", "method", "ate_m", "rpe_m", "loops", "ms_per_frame"]]))
    if not semantic:
        return df
    det = make_detector(TUM_CLASSES, **(detector_args or {}))
    frames = list(range(0, len(tum), 5))
    with GPUMonitor() as mon:
        cache, det_ms = CMP.cache_detections(tum, det, frames)
    a = ate(P, tum.gt_poses); Pw = [a["T_align"] @ T for T in P]
    o = run_mapping(tum, Pw, S, None, TUM_CLASSES, gasf_stride=5, tsdf_voxel=0.01, tsdf_depth=3.0, max_depth=4.0,
                    use_gt_eval=False, det_cache=cache, seg_kwargs=dict(min_score=0.30), verbose=False)
    twin = DigitalTwin("TUM office twin"); twin.update(o["gasf"].finalize(), TUM_CLASSES, "tum-office", occ_new=o["occ"])
    pc = o["tsdf"].point_cloud()
    build_dashboard(O.root / "TUM_Dashboard.html", twin, np.asarray(pc.points), np.asarray(pc.colors),
                    traj_est=[T[:3, 3] for T in Pw], traj_gt=[T[:3, 3] for T in tum.gt_poses],
                    perf=dict(ate_cm=100 * a["rmse"], stages_ms={"Detector + SAM 2.1": det_ms, **o["timings_ms"]},
                              details={"GPU power (mean)": f"{mon.stats().get('power_mean_w', float('nan')):.0f} W"}),
                    title="GeoSemTwin-X · TUM office twin", subtitle=s0)
    O.merge_json("summary", {"tum": dict(sequences=list(seqs), slam=df.to_dict("records"), assets=twin.counts(),
                                         detector_ms=det_ms, gpu=mon.stats())})
    return df


# =============================================================================== Staer
def _photometric_error(seq, gap=3, every=10):
    """Warp frame i+gap into frame i with the ground-truth poses and depth; median grey-level error (0-255).
    Small (a few levels) when RGB, depth and poses refer to the same instants; large when they are shifted."""
    errs = []
    K = seq.K
    for i in range(5, len(seq) - gap, every):
        a, b = seq.frame(i), seq.frame(i + gap)
        from .mapping import backproject
        Pw, z, (v, u) = backproject(a["depth"], K, seq.gt_poses[i], stride=2, max_depth=15.0)
        T = np.linalg.inv(seq.gt_poses[i + gap]); Pc = Pw @ T[:3, :3].T + T[:3, 3]
        ok = Pc[:, 2] > 0.2
        uu = np.round(K[0, 0] * Pc[ok, 0] / Pc[ok, 2] + K[0, 2]).astype(int)
        vv = np.round(K[1, 1] * Pc[ok, 1] / Pc[ok, 2] + K[1, 2]).astype(int)
        h, w = a["depth"].shape
        m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        ga = a["rgb"].mean(-1)[v[ok][m], u[ok][m]]; gb = b["rgb"].mean(-1)[vv[m], uu[m]]
        if m.sum() > 500:
            errs.append(float(np.median(np.abs(ga - gb))))
    return float(np.median(errs)) if errs else float("nan")


def diagnose_staer(scene="scene03", walk="001", root="data/staer"):
    """Checks that RGB, depth, instance masks and poses of a Staer walkthrough line up frame by frame,
    comparing ffmpeg's default (constant-frame-rate) decoding with frame-exact passthrough decoding."""
    from .data import StaerSequence
    res = {}
    for mode in ("default", "passthrough"):
        print(f"\n--- decoding with ffmpeg {mode} mode ---")
        st = StaerSequence(scene, walk, root=root, scale=0.25, stride=1, sync=mode)
        res[mode] = (st.decoded_frames, len(st.gt_poses), _photometric_error(st))
    print("\n==== Staer stream alignment ====")
    for mode, (cnt, n, e) in res.items():
        print(f"{mode:12s} decoded frames {cnt}  poses {n}  photometric error (grey levels, lower = aligned) {e:.1f}")
    d, p = res["default"][2], res["passthrough"][2]
    if p < 0.7 * d:
        print("=> default decoding was misaligned; passthrough fixes it. Re-run vigs-export, VIGS-SLAM and staer.")
    elif abs(p - d) <= 0.3 * max(d, 1e-6):
        print("=> both modes agree; frame alignment is not the problem.")
    else:
        print("=> results inconclusive; send this output.")


def run_staer(out, scene="scene03", walk="001", root="data/staer", vigs_traj=None, detector_args=None,
              det_every=3, sweep=True, change_walk=None, quick=False):
    from .data import StaerSequence
    O = Out(out); summary = {}
    st = StaerSequence(scene, walk, root=root, scale=0.5, stride=2)
    print(f"Staer {scene}/{walk}: {len(st)} frames, GT assets", pd.Series(list(st.instance_class.values())).value_counts().to_dict())
    # ---------------- SLAM: configuration sweep -> full run --------------------------------------------------------
    sweep_cfgs = {"hybrid, depth ≤ 6 m, stride 1": ("s1", dict(tracker="hybrid", depth_max=6.0)),
                  "hybrid, depth ≤ 10 m, stride 2": ("s2", dict(tracker="hybrid", depth_max=10.0)),
                  "hybrid, depth ≤ 6 m, stride 2": ("s2", dict(tracker="hybrid", depth_max=6.0)),
                  "hybrid + ICP, depth ≤ 8 m, stride 2": ("s2", dict(tracker="hybrid+icp", depth_max=8.0))}
    if sweep:
        short = {"s1": StaerSequence(scene, walk, root=root, scale=0.5, stride=1, max_frames=120),
                 "s2": StaerSequence(scene, walk, root=root, scale=0.5, stride=2, max_frames=60)}
        rows = [CMP.slam_study({"first 6 s": short[s]}, {n: kw}, derived=False) for n, (s, kw) in sweep_cfgs.items()]
        df_sw = pd.concat(rows, ignore_index=True); O.save("staer_slam_sweep", df_sw)
        best = df_sw.sort_values("ate_m").iloc[0]["method"]
        O.add_study("staer_slam_sweep", "SLAM configuration sweep (Staer, first 6 s)", f"best: {best}",
                    df_sw[["method", "ate_m", "rpe_m", "fallbacks", "ms_per_frame"]], highlight=best)
    else:
        best = "hybrid, depth ≤ 6 m, stride 1"
    s_key, kw = sweep_cfgs[best]
    st_slam = st if s_key == "s2" else StaerSequence(scene, walk, root=root, scale=0.5, stride=1)
    t0 = time.time(); slam, P = run_slam(st_slam, depth_max=kw["depth_max"], tracker=kw["tracker"], verbose=True)
    slam_ms = 1000 * (time.time() - t0) / len(st_slam)
    P_ours, S_ours, P_odo = P, slam.step_sigmas, slam.poses_before_opt
    if s_key == "s1":
        P_ours, S_ours, P_odo = P[::2][:len(st)], list(S_ours)[::2][:len(st)], P_odo[::2][:len(st)]
    trajs = {"GeoSemTwin-X RGB-D SLAM": P_ours, "  w/o loop closure": P_odo}
    sig = {"GeoSemTwin-X RGB-D SLAM": S_ours, "  w/o loop closure": S_ours}
    if vigs_traj:
        from .vigs_bridge import load_vigs_trajectory
        P_v, fname = load_vigs_trajectory(vigs_traj, st.timestamps)
        trajs[f"VIGS-SLAM mono+IMU ({fname})"] = P_v; sig[f"VIGS-SLAM mono+IMU ({fname})"] = None
    rows = []
    for name, Pt in trajs.items():
        a, r = ate(Pt, st.gt_poses), rpe(Pt, st.gt_poses, st.timestamps)
        s_, R_, t_ = umeyama(np.array([T[:3, 3] for T in Pt]), np.array([T[:3, 3] for T in st.gt_poses]), True)
        rows.append(dict(method=name.strip(), ate_m=a["rmse"], ate_sim3_scale=s_, rpe_m=r["trans_rmse"],
                         rpe_deg=r["rot_rmse_deg"], drift_pct=100 * a["rmse"] / path_length(st.gt_poses),
                         ms_per_frame=slam_ms if "GeoSemTwin" in name else np.nan))
    df_sl = pd.DataFrame(rows); O.save("staer_slam", df_sl); print(CMP.table_md(df_sl))
    plot_trajectories(O.fig / "staer_trajectories.png", st.gt_poses, {k.strip(): v for k, v in trajs.items()},
                      f"Staer {scene}/{walk}: trajectories (top view)")
    O.add_study("staer_slam", "SLAM on the Staer warehouse: RGB-D SLAM vs VIGS-SLAM",
                "ATE after SE(3) alignment; ate_sim3_scale = scale needed (1.0 = metric)", df_sl,
                O.fig / "staer_trajectories.png", highlight="GeoSemTwin-X RGB-D SLAM")
    summary["slam"] = df_sl.to_dict("records")
    # ---------------- detections on every frame (reused everywhere) + latency / power -----------------------------
    det = make_detector(STAER_CLASSES, **(detector_args or {}))
    idle = idle_power(3.0)
    all_frames = list(range(len(st))) if not quick else list(range(0, len(st), det_every))
    with GPUMonitor() as mon:
        cache, det_ms = CMP.cache_detections(st, det, all_frames)
    gpu = mon.stats()
    CMP.save_cache(O.root / "staer_detections.npz", cache)
    frames = list(range(0, len(st), det_every))
    seg_kw = dict(min_score=0.30, floor_classes=FLOOR_CLASSES)
    vis = visible_gt_assets(st, frames, max_depth=8.0)
    print(f"{sum(len(v) for v in cache.values())} detections, {det_ms:.0f} ms/frame on GPU; {len(vis)} detectable assets")
    # ---------------- pose sets ----------------------------------------------------------------------------------
    pose_sets = {"ground-truth poses": (st.gt_poses, None)}
    for name, Pt in trajs.items():
        if "w/o loop" in name:
            continue
        a = ate(Pt, st.gt_poses)
        pose_sets[name.strip()] = ([a["T_align"] @ T for T in Pt], sig[name])
    # ---------------- fusion vs conventional + ablations -----------------------------------------------------------
    df_f, _ = CMP.fusion_study(st, cache, frames, STAER_CLASSES, pose_sets, seg_kwargs=seg_kw, max_depth=8.0)
    O.save("staer_fusion", df_f)
    cols = ["poses", "method", "objects", "precision", "recall", "f1", "duplicate_rate", "identity_consistency",
            "id_switches", "label_acc", "miou"]
    print(CMP.table_md(df_f[cols]))
    g = df_f[df_f.poses == "ground-truth poses"]
    CMP.plot_methods(g, O.fig / "staer_methods.png", title="Semantic fusion vs conventional methods (Staer, GT poses)")
    CMP.plot_ablation(g, O.fig / "staer_ablation.png", "f1", "F1: every method and GASF ablation (Staer)")
    CMP.plot_ablation(g, O.fig / "staer_ablation_idc.png", "identity_consistency", "Identity consistency (Staer)")
    O.add_study("staer_fusion", "Semantic fusion vs conventional methods (Staer)",
                f"{len(vis)} detectable assets · identical detections for all methods · rows per pose source",
                df_f[cols], O.fig / "staer_methods.png")
    O.add_study("staer_ablation", "GASF ablation: F1", "each row removes one GASF component", None, O.fig / "staer_ablation.png")
    O.add_study("staer_ablation_idc", "GASF ablation: identity consistency", "", None, O.fig / "staer_ablation_idc.png")
    summary["fusion"] = df_f.to_dict("records")
    # ---------------- robustness / sensitivity ----------------------------------------------------------------------
    df_n = CMP.pose_noise_study(st, cache, frames, STAER_CLASSES, levels=(0.0, 0.05, 0.1, 0.2, 0.4),
                                seeds=(0, 1) if quick else (0, 1, 2), seg_kwargs=seg_kw)
    O.save("staer_pose_noise", df_n)
    CMP.plot_lines(df_n, "pose_error_m", "f1", O.fig / "noise_f1.png", "F1 vs pose drift", "end-of-run pose drift (m)", "F1")
    CMP.plot_lines(df_n, "pose_error_m", "duplicate_rate", O.fig / "noise_dup.png", "Duplicates per asset vs pose drift",
                   "end-of-run pose drift (m)", "duplicates per asset")
    O.add_study("noise_f1", "Robustness to pose drift (F1)", "random-walk drift added to GT poses", None, O.fig / "noise_f1.png")
    O.add_study("noise_dup", "Robustness to pose drift (duplicates)", "lower is better",
                df_n[["pose_error_m", "method", "f1", "identity_consistency", "duplicate_rate"]], O.fig / "noise_dup.png")
    df_t = CMP.threshold_study(st, cache, frames, STAER_CLASSES, thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
                               seg_kwargs=dict(floor_classes=FLOOR_CLASSES))
    O.save("staer_threshold", df_t)
    CMP.plot_lines(df_t, "threshold", "f1", O.fig / "threshold_f1.png", "F1 vs detector threshold", "box score threshold", "F1")
    O.add_study("threshold", "Sensitivity to the detector threshold", "GT poses",
                df_t[["threshold", "method", "precision", "recall", "f1"]], O.fig / "threshold_f1.png")
    other_ms = float(np.mean([slam_ms]))
    if not quick:
        df_k = CMP.keyframe_interval_study(st, cache, STAER_CLASSES, intervals=(1, 2, 3, 5, 10), seg_kwargs=seg_kw,
                                           det_ms=det_ms, other_ms=other_ms)
        O.save("staer_keyframe_interval", df_k)
        CMP.plot_lines(df_k, "interval", "f1", O.fig / "keyframe_f1.png", "F1 vs detector interval",
                       "run the detector every k frames", "F1")
        O.add_study("keyframe", "How often must the detector run?",
                    f"detector {det_ms:.0f} ms/frame on GPU; cost per input frame = det_ms / k",
                    df_k[["interval", "method", "f1", "identity_consistency", "detector_ms_per_frame"]], O.fig / "keyframe_f1.png")
    df_m = CMP.map_quality_study(st, {k: v for k, v in pose_sets.items()}, frames=list(range(0, len(st), 2)), voxel=0.05)
    O.save("staer_map_quality", df_m); print(CMP.table_md(df_m))
    O.add_study("map_quality", "Reconstruction quality: uncertainty-aware vs naive fusion",
                "accuracy = map-to-GT distance; completeness = GT surface within 5 cm", df_m,
                highlight="Uncertainty-aware fusion (ours)")
    # ---------------- latency / power table -------------------------------------------------------------------------
    o = run_mapping(st, st.gt_poses, None, None, STAER_CLASSES, gasf_stride=det_every, tsdf_voxel=0.05, tsdf_depth=10.0,
                    max_depth=8.0, det_cache=cache, seg_kwargs=seg_kw, map_sigma_max=0.15, verbose=False)
    lat = pd.DataFrame([
        dict(stage="Grounding DINO + SAM 2.1 (GPU)", unit="per keyframe", ms=det_ms),
        dict(stage="RGB-D SLAM tracking (CPU)", unit="per frame", ms=slam_ms),
        dict(stage="Fused map integration", unit="per frame", ms=o["timings_ms"]["tsdf"]),
        dict(stage="Occupancy update", unit="per keyframe", ms=o["timings_ms"]["occ"]),
        dict(stage="3D lifting of detections", unit="per keyframe", ms=o["timings_ms"]["lift"]),
        dict(stage="GASF association + fusion", unit="per keyframe", ms=o["timings_ms"]["fuse"])])
    fps = 1000.0 / (slam_ms + o["timings_ms"]["tsdf"] + (det_ms + o["timings_ms"]["occ"] + o["timings_ms"]["lift"]
                                                           + o["timings_ms"]["fuse"]) / det_every)
    power = dict(gpu_power_mean_w=gpu.get("power_mean_w"), gpu_power_peak_w=gpu.get("power_peak_w"),
                 gpu_idle_w=idle.get("power_mean_w"), gpu_util_pct=gpu.get("util_mean_pct"),
                 gpu_mem_peak_mb=gpu.get("mem_peak_mb"),
                 energy_per_keyframe_j=(gpu.get("power_mean_w", np.nan) * det_ms / 1000.0) if gpu.get("available") else None,
                 end_to_end_fps=fps, ram_peak_gb=o["peak_ram_gb"])
    O.save("staer_latency", lat); O.merge_json("power", power)
    O.add_study("latency", "Latency per stage and compute power",
                f"end-to-end {fps:.2f} FPS · GPU {power['gpu_power_mean_w'] or float('nan'):.0f} W while detecting "
                f"(idle {power['gpu_idle_w'] or float('nan'):.0f} W) · peak GPU memory {power['gpu_mem_peak_mb'] or 0:.0f} MB", lat,
                highlight="")
    summary.update(latency=lat.to_dict("records"), power=power)
    # ---------------- change detection: second walkthrough of the same static scene ---------------------------------
    twin = DigitalTwin(f"Staer {scene} twin"); store = TwinStore(O.root / "twin_store")
    twin.update(o["gasf"].finalize(), STAER_CLASSES, f"{scene}-{walk}", occ_new=o["occ"]); store.push(twin)
    if change_walk:
        st2 = StaerSequence(scene, change_walk, root=root, scale=0.5, stride=2)
        fr2 = list(range(0, len(st2), det_every))
        cache2, _ = CMP.cache_detections(st2, det, fr2)
        o2 = run_mapping(st2, st2.gt_poses, None, None, STAER_CLASSES, gasf_stride=det_every, tsdf_voxel=0.05,
                         tsdf_depth=10.0, max_depth=8.0, det_cache=cache2, seg_kwargs=seg_kw, map_sigma_max=0.15, verbose=False)
        v, log = twin.update(o2["gasf"].finalize(), STAER_CLASSES, f"{scene}-{change_walk}", occ_new=o2["occ"])
        store.push(twin)
        reported = [c for c in log if c["type"] in ("moved", "removed", "added", "geometry")]
        summary["static_scene_change"] = dict(changes_reported=len(reported), unverified=len([c for c in log if c["type"] == "unverified"]),
                                              assets_compared=len(twin.present()))
        print("static scene (no real changes) -> false alarms:", len(reported))
    # ---------------- dashboard ----------------------------------------------------------------------------------
    pc = o["tsdf"].point_cloud()
    P_main = pose_sets.get("GeoSemTwin-X RGB-D SLAM", (st.gt_poses, None))[0]
    build_dashboard(O.root / "GeoSemTwinX_Dashboard.html", twin, np.asarray(pc.points), np.asarray(pc.colors),
                    traj_est=[T[:3, 3] for T in P_main], traj_gt=[T[:3, 3] for T in st.gt_poses], studies=O.studies(),
                    perf=dict(ate_cm=100 * float(df_sl.ate_m.iloc[0]),
                              stages_ms={f"{r.stage} ({r.unit})": r.ms for r in lat.itertuples()},
                              details={k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in power.items()}),
                    title="GeoSemTwin-X Digital Twin", subtitle=f"Staer warehouse {scene} / walkthrough {walk}")
    summary["twin"] = dict(assets=twin.counts(), n=len(twin.present()), versions=twin.version)
    O.merge_json("summary", {"staer": summary})
    print("dashboard:", O.root / "GeoSemTwinX_Dashboard.html")
    return summary


# =============================================================================== synthetic (exact GT)
def run_synthetic(out, seeds=(0, 1, 2), budget=80.0, skip_slam=False):
    from . import bench
    O = Out(out); bench.set_out(O.root / "synthetic")
    if not skip_slam:
        bench.exp_slam()
    bench.exp_semantic(seeds=seeds)
    bench.exp_change(seeds=seeds)
    bench.exp_nbv(seeds=seeds, budget=budget)
    R = O.root / "synthetic"
    # fusion
    sem = pd.DataFrame(json.loads((R / "semantic_slam.json").read_text()) if (R / "semantic_slam.json").exists()
                       else json.loads((R / "semantic_gtpose.json").read_text()))
    sem_m = sem.groupby("method", as_index=False).mean(numeric_only=True)
    O.save("synthetic_fusion", sem_m)
    CMP.plot_methods(sem, O.fig / "synthetic_methods.png", title="Synthetic warehouse: fusion vs conventional (SLAM poses)")
    O.add_study("synthetic_fusion", "Fusion vs conventional (synthetic, exact GT, SLAM poses)",
                f"mean of {len(seeds)} detector-noise seeds", sem_m[["method", "objects", "precision", "recall", "f1",
                                                                     "duplicate_rate", "identity_consistency", "miou"]],
                O.fig / "synthetic_methods.png")
    # change detection
    ch = json.loads((R / "change.json").read_text())
    dch = pd.DataFrame([dict(method="Occupancy-verified change detection (ours)", precision=np.mean([c["precision"] for c in ch]),
                             recall=np.mean([c["recall"] for c in ch]), f1=np.mean([c["f1"] for c in ch])),
                        dict(method="Without occupancy verification", precision=np.mean([c["no_visibility"]["precision"] for c in ch]),
                             recall=np.mean([c["no_visibility"]["recall"] for c in ch]), f1=np.mean([c["no_visibility"]["f1"] for c in ch]))])
    O.save("synthetic_change", dch)
    O.add_study("change", "Change detection between two mapping sessions",
                "ground truth: 2 moved, 2 removed, 2 added assets", dch, highlight="Occupancy-verified change detection (ours)")
    # exploration
    nbv = json.loads((R / "nbv.json").read_text())
    grid = np.arange(0, budget + 1, 5)
    rows, curves = [], {}
    names = {"nbv": "NBV (ours: unknown + uncertainty + ambiguity)", "nbv_no_semantic": "NBV w/o ambiguity term",
             "nbv_coverage_only": "NBV coverage only", "frontier": "Nearest frontier", "random": "Random goals"}
    for st in names:
        runs = [v for v in nbv.values() if v["strategy"] == st]
        if not runs:
            continue
        cv = [np.interp(grid, [r["path_m"] for r in v["log"]], [r["coverage"] for r in v["log"]]) for v in runs]
        curves[names[st]] = np.mean(cv, 0)
        half = min(40.0, budget / 2)
        rows.append(dict(method=names[st], **{f"coverage_at_{half:.0f}m": float(np.mean([np.interp(half, grid, c) for c in cv]))},
                         coverage_final=float(np.mean([c[-1] for c in cv])),
                         assets_identified_pct=100 * float(np.mean([v["assets_identified"] / v["total_assets"] for v in runs])),
                         replans=float(np.mean([v["log"][-1]["replans"] for v in runs])),
                         collisions=float(np.mean([v["log"][-1]["collisions"] for v in runs]))))
    dn = pd.DataFrame(rows); O.save("synthetic_exploration", dn)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    cols = ["#1F5BD8", "#7FA3EA", "#BACDF3", "#D9731A", "#A04FB0"]
    for (n, c), col in zip(curves.items(), cols):
        ax.plot(grid, 100 * c, lw=2.6 if "ours" in n else 1.6, color=col, label=n)
    ax.set_xlabel("path length (m)"); CMP._style(ax, "Exploration: surface coverage vs distance", "coverage (%)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(O.fig / "exploration.png", dpi=160); plt.close(fig)
    O.add_study("exploration", "Exploration strategies (simulated robot, LiDAR + RGB-D)", f"mean of {len(seeds)} start poses",
                dn, O.fig / "exploration.png", highlight=names["nbv"])
    O.merge_json("summary", {"synthetic": dict(fusion=sem_m.to_dict("records"), change=dch.to_dict("records"),
                                               exploration=dn.to_dict("records"))})


# =============================================================================== report
def write_report(out):
    O = Out(out)
    S = json.loads((O.root / "summary.json").read_text()) if (O.root / "summary.json").exists() else {}
    L = ["# GeoSemTwin-X — experiment summary", ""]
    for name in sorted(p.stem for p in O.cmp.glob("*.csv") if not p.stem.endswith("_card")):
        df = pd.read_csv(O.cmp / f"{name}.csv")
        L += [f"## {name.replace('_', ' ')}", "", CMP.table_md(df), ""]
    L += ["## Key findings (computed from the tables above)", ""]
    try:
        f = pd.DataFrame(S["staer"]["fusion"]); g = f[f.poses == "ground-truth poses"].set_index("method")
        L.append(f"- Duplicates per asset: per-frame 2D labelling {g.loc['Independent 2D labelling','duplicate_rate']:.2f} → "
                 f"GASF {g.loc['GASF (ours)','duplicate_rate']:.2f}; objects in map {g.loc['Independent 2D labelling','objects']:.0f} → "
                 f"{g.loc['GASF (ours)','objects']:.0f}.")
        best_base = g.drop([m for m in g.index if m.startswith("GASF")])["f1"].idxmax()
        L.append(f"- F1: GASF {g.loc['GASF (ours)','f1']:.2f} vs best conventional ({best_base}) {g.loc[best_base,'f1']:.2f}.")
        L.append(f"- Identity consistency: GASF {g.loc['GASF (ours)','identity_consistency']:.2f} vs per-frame "
                 f"{g.loc['Independent 2D labelling','identity_consistency']:.2f}.")
    except Exception:
        pass
    try:
        for r in S["staer"]["slam"]:
            L.append(f"- Staer SLAM {r['method']}: ATE {100 * r['ate_m']:.1f} cm ({r['drift_pct']:.2f}% drift).")
        p = S["staer"]["power"]
        L.append(f"- Compute: {p['end_to_end_fps']:.2f} FPS end-to-end, GPU {p['gpu_power_mean_w']:.0f} W while detecting, "
                 f"peak GPU memory {p['gpu_mem_peak_mb']:.0f} MB.")
    except Exception:
        pass
    (O.root / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L[-8:]))
    print("report:", O.root / "REPORT.md")
