#!/usr/bin/env python3
"""GeoSemTwin-X — one entry point for the prototype, the comparative studies and the demo.

  python scripts/run_all.py check                      # environment check (GPU, packages, ffmpeg, HF login, ROS, VIGS)
  python scripts/run_all.py tum   [--all-seqs]         # SLAM vs published baselines + ablations, office twin
  python scripts/run_all.py vigs-export                # Staer walkthrough -> VIGS-SLAM input (images, imu.csv, calib, yaml)
  python scripts/run_all.py vigs-run --vigs-root ~/VIGS-SLAM --vigs-env vigs-slam-5090
  python scripts/run_all.py staer [--vigs-traj results/vigs_out]   # all Staer comparative studies + dashboard
  python scripts/run_all.py synthetic                  # exact-GT: fusion, change detection, exploration (CPU)
  python scripts/run_all.py demo  [--source staer|tum] # live demo video + dashboard
  python scripts/run_all.py report                     # results/REPORT.md with every table + key findings
  python scripts/run_all.py all   [--vigs-traj ...]    # tum + staer + synthetic + report
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def check(args):
    import importlib
    import shutil
    print("Python", sys.version.split()[0], "| cwd", ROOT)
    for m in ["numpy", "scipy", "pandas", "cv2", "open3d", "plotly", "matplotlib", "torch", "transformers", "sam2",
              "huggingface_hub", "psutil", "tabulate"]:
        try:
            mod = importlib.import_module(m); print(f"  ok   {m:16s} {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"  MISS {m:16s} {e}")
    try:
        import torch
        print("  CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB, bf16: {torch.cuda.is_bf16_supported()}")
    except Exception:
        pass
    print("  ffmpeg:", shutil.which("ffmpeg") or "MISSING (needed for the Staer dataset)")
    from geosemtwinx.power import _smi
    print("  nvidia-smi:", _smi() or "missing (power measurements disabled)")
    try:
        from huggingface_hub import whoami
        print("  Hugging Face login:", whoami()["name"])
    except Exception as e:
        print("  Hugging Face login: NOT logged in -> run `huggingface-cli login`", type(e).__name__)
    try:
        import rclpy  # noqa
        print("  ROS 2 rclpy: ok (", os.environ.get("ROS_DISTRO", "?"), ")")
    except Exception:
        print("  ROS 2 rclpy: not importable in this Python (source /opt/ros/<distro>/setup.bash first)")
    if args.vigs_root:
        print("  VIGS-SLAM:", "found" if (Path(args.vigs_root) / "demo.py").exists() else "demo.py NOT found", args.vigs_root)


def det_args(a):
    return dict(dino=a.dino, sam=a.sam, threshold=0.20)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "diagnose", "tum", "vigs-export", "vigs-run", "staer", "synthetic", "demo", "report", "all"])
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--dino", default="tiny", help="grounding DINO size: tiny | base")
    p.add_argument("--sam", default="tiny", help="SAM 2.1 size: tiny | small | base_plus")
    p.add_argument("--all-seqs", action="store_true", help="TUM: fr1/desk + fr2/xyz + fr3/long_office_household")
    p.add_argument("--with-icp", action="store_true", help="TUM: also run the (slow) ICP-only tracker ablation")
    p.add_argument("--scene", default="scene03"); p.add_argument("--walk", default="001")
    p.add_argument("--change-walk", default=None, help="second walkthrough of the same scene, e.g. 002 (static-scene change test)")
    p.add_argument("--vigs-root", default=os.environ.get("VIGS_ROOT"), help="path to your VIGS-SLAM clone")
    p.add_argument("--vigs-env", default=None, help="conda env of VIGS-SLAM (e.g. vigs-slam-5090); omit to use --vigs-python")
    p.add_argument("--vigs-python", default="python")
    p.add_argument("--vigs-gsmapping", action="store_true")
    p.add_argument("--vigs-traj", default=None, help="VIGS-SLAM output folder (or trajectory file) to compare / use")
    p.add_argument("--det-every", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="fewer seeds / skip the keyframe-interval study")
    p.add_argument("--seeds", type=int, default=3); p.add_argument("--budget", type=float, default=80.0)
    p.add_argument("--source", default="staer", choices=["staer", "tum"]); p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args()
    t0 = time.time()
    from geosemtwinx import experiments as E

    if a.cmd == "check":
        check(a)
    if a.cmd == "diagnose":
        E.diagnose_staer(a.scene, a.walk, root=f"{a.data}/staer")
    if a.cmd in ("tum", "all"):
        seqs = ["rgbd_dataset_freiburg1_desk"] + (["rgbd_dataset_freiburg2_xyz", "rgbd_dataset_freiburg3_long_office_household"]
                                                  if a.all_seqs else [])
        cfgs = ["Full (hybrid + PnP + loops)", "w/o PnP recovery"] + (["ICP tracker (geometry only)"] if a.with_icp else [])
        E.run_tum(a.out, seqs, cfgs, root=f"{a.data}/tum", detector_args=det_args(a))
    if a.cmd == "vigs-export":
        from geosemtwinx.vigs_bridge import export_staer, vigs_command
        d = export_staer(a.scene, a.walk, out_dir=f"{a.data}/vigs_{a.scene}_{a.walk}", root=f"{a.data}/staer",
                         vigs_root=a.vigs_root)
        print("\nRun VIGS-SLAM (from your VIGS-SLAM folder, in its environment):\n ",
              " ".join(vigs_command(a.vigs_root or ".", d, f"{a.out}/vigs_{a.scene}_{a.walk}", gsmapping=a.vigs_gsmapping)))
    if a.cmd == "vigs-run":
        from geosemtwinx.vigs_bridge import run_vigs
        assert a.vigs_root, "--vigs-root is required"
        run_vigs(a.vigs_root, f"{a.data}/vigs_{a.scene}_{a.walk}", f"{a.out}/vigs_{a.scene}_{a.walk}", conda_env=a.vigs_env,
                 python=a.vigs_python, gsmapping=a.vigs_gsmapping)
    if a.cmd in ("staer", "all"):
        vt = a.vigs_traj
        if vt is None and Path(f"{a.out}/vigs_{a.scene}_{a.walk}").exists():
            vt = f"{a.out}/vigs_{a.scene}_{a.walk}"
            print("using VIGS-SLAM output found in", vt)
        E.run_staer(a.out, a.scene, a.walk, root=f"{a.data}/staer", vigs_traj=vt, detector_args=det_args(a),
                    det_every=a.det_every, change_walk=a.change_walk, quick=a.quick)
    if a.cmd in ("synthetic", "all"):
        E.run_synthetic(a.out, seeds=tuple(range(2 if a.quick else a.seeds)), budget=a.budget)
    if a.cmd == "demo":
        from geosemtwinx.demo import run_demo
        if a.source == "staer":
            from geosemtwinx.data import StaerSequence
            seq = StaerSequence(a.scene, a.walk, root=f"{a.data}/staer", scale=0.5, stride=1, max_frames=a.max_frames)
            det = E.make_detector(E.STAER_CLASSES, **det_args(a)); det.bt = 0.30
            run_demo(seq, E.STAER_CLASSES, det, f"{a.out}/demo_staer", T_init=seq.gt_poses[0], detect_every=a.det_every,
                     depth_max=6.0, floor_classes=E.FLOOR_CLASSES, gt_traj=[T[:3, 3] for T in seq.gt_poses],
                     title="GeoSemTwin-X live demo — Staer warehouse")
        else:
            from geosemtwinx.data import TUMSequence
            seq = TUMSequence("rgbd_dataset_freiburg1_desk", root=f"{a.data}/tum", max_frames=a.max_frames)
            det = E.make_detector(E.TUM_CLASSES, **det_args(a)); det.bt = 0.30
            run_demo(seq, E.TUM_CLASSES, det, f"{a.out}/demo_tum", T_init=None, detect_every=a.det_every, depth_max=4.0,
                     title="GeoSemTwin-X live demo — TUM office")
    if a.cmd in ("report", "all"):
        E.write_report(a.out)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
