"""Offline live demo: stream a dataset through LiveTwin and record a side-by-side video
(camera view with detections + asset IDs | top-down live twin with map, assets, path and next-best view)."""
import time
from pathlib import Path

import cv2
import numpy as np

from .streaming import LiveTwin

PALETTE = [(216, 91, 31), (26, 115, 217), (127, 157, 42), (176, 79, 160), (90, 59, 194), (31, 127, 91)]   # BGR


def _color(classes, label):
    return PALETTE[classes.index(label) % len(PALETTE)] if label in classes else (128, 128, 128)


def draw_assets(im, assets, classes, T_wc, K):
    """Project the persistent 3D assets of the twin into the current view (ID + class)."""
    Tcw = np.linalg.inv(T_wc); h, w = im.shape[:2]
    for a in assets:
        lo, hi = np.asarray(a["lo"]), np.asarray(a["hi"])
        C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        Pc = C @ Tcw[:3, :3].T + Tcw[:3, 3]
        if (Pc[:, 2] < 0.3).any():
            continue
        u = K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2]; v = K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2]
        x0, x1, y0, y1 = int(u.min()), int(u.max()), int(v.min()), int(v.max())
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        c = _color(classes, a["label"])
        cv2.rectangle(im, (x0, y0), (x1, y1), c, 1, cv2.LINE_AA)
        txt = f'A{a["id"]:02d} {a["label"]}'
        cv2.putText(im, txt, (max(x0, 2), min(max(y1 + 14, 14), h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(im, txt, (max(x0, 2), min(max(y1 + 14, 14), h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return im


def draw_camera(rgb, dets, classes, info, assets=None, T_wc=None, K=None):
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    over = im.copy()
    for d in dets:
        if "mask" in d:
            over[d["mask"]] = _color(classes, d["label"])
    im = cv2.addWeighted(over, 0.35, im, 0.65, 0)
    for d in dets:
        if "mask" not in d:
            continue
        ys, xs = np.nonzero(d["mask"])
        if len(xs) == 0:
            continue
        c = _color(classes, d["label"])
        cv2.rectangle(im, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), c, 2)
        cv2.putText(im, f'{d["label"]} {d["score"]:.2f}', (int(xs.min()), max(int(ys.min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(im, f'{d["label"]} {d["score"]:.2f}', (int(xs.min()), max(int(ys.min()) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    if assets is not None and T_wc is not None:
        draw_assets(im, assets, classes, T_wc, K)
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (20, 20, 20), -1)
    cv2.putText(im, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return im


def draw_topdown(lt, size=540, span=None, nbv=None, gt_traj=None):
    P, C = lt.map_points(60000)
    traj = np.array(lt.traj)
    img = np.full((size, size, 3), 250, np.uint8)
    if len(traj) == 0:
        return img
    c = traj[-1, :2]
    span = span or 16.0
    s = size / span

    def px(xy):
        q = (np.asarray(xy)[..., :2] - c) * s
        return np.stack([size / 2 + q[..., 0], size / 2 - q[..., 1]], -1).astype(int)

    if len(P):
        z = P[:, 2]
        keep = z < lt.floor_z + 2.2
        q = px(P[keep]); ok = (q[:, 0] >= 0) & (q[:, 0] < size) & (q[:, 1] >= 0) & (q[:, 1] < size)
        col = (C[keep][ok][:, ::-1] * 160 + 80).astype(np.uint8)
        img[q[ok, 1], q[ok, 0]] = col
    if gt_traj is not None and len(gt_traj):
        cv2.polylines(img, [px(np.asarray(gt_traj)[: len(traj)])], False, (140, 160, 40), 1, cv2.LINE_AA)
    cv2.polylines(img, [px(traj)], False, (31, 115, 217), 2, cv2.LINE_AA)
    for a in lt.assets():
        p0, p1 = px(a["lo"]), px(a["hi"])
        col = _color(lt.classes, a["label"])
        cv2.rectangle(img, (min(p0[0], p1[0]), min(p0[1], p1[1])), (max(p0[0], p1[0]), max(p0[1], p1[1])), col, 2)
        cv2.putText(img, f'A{a["id"]:02d}', (max(p0[0], p1[0]) + 2, min(p0[1], p1[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (40, 40, 40), 1, cv2.LINE_AA)
    if nbv and "goal" in nbv:
        g = px(np.array(nbv["goal"])); path = px(np.array(nbv["path"]))
        cv2.polylines(img, [path], False, (0, 160, 255), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(g), 7, (0, 160, 255), 2)
        e = g + (14 * np.array([np.cos(nbv["yaw"]), -np.sin(nbv["yaw"])])).astype(int)
        cv2.arrowedLine(img, tuple(g), tuple(e), (0, 160, 255), 2, tipLength=0.4)
    cv2.circle(img, (size // 2, size // 2), 5, (31, 115, 217), -1)
    cv2.rectangle(img, (0, 0), (size, 26), (20, 20, 20), -1)
    cv2.putText(img, f"live twin: {len(lt.assets())} assets  |  orange = next-best view", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def run_demo(seq, classes, detector, out_dir, T_init=None, detect_every=3, depth_max=6.0, floor_z=0.0,
             floor_classes=(), max_frames=None, nbv_every=15, gt_traj=None, title="GeoSemTwin-X live demo", fps=10):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f0 = seq.frame(0); h, w = f0["depth"].shape
    lt = LiveTwin(seq.K, w, h, classes, detector=detector, T_init=T_init, detect_every=detect_every,
                  depth_max=depth_max, floor_z=floor_z, floor_classes=floor_classes, nbv=T_init is not None)
    n = min(len(seq), max_frames or len(seq))
    vw, nbv, last_dets = None, None, []
    t0 = time.time()
    for i in range(n):
        f = seq.frame(i)
        r = lt.process(f)
        last_dets = r["detections"] if r["keyframe"] else []        # 2D detections only on the frame they came from
        if T_init is not None and i % nbv_every == 0 and i > 0:
            nbv = lt.plan_next_view()
        info = (f"frame {i}/{n}  {'KEYFRAME: ' + str(len(last_dets)) + ' detections' if r['keyframe'] else 'tracking'}"
                f"  |  boxes = persistent twin assets  |  {(i + 1) / (time.time() - t0):.1f} FPS")
        cam = draw_camera(f["rgb"], last_dets, classes, info, assets=lt.assets(), T_wc=r["pose"], K=lt.K)
        top = draw_topdown(lt, size=cam.shape[0], nbv=nbv, gt_traj=gt_traj)
        frame = np.hstack([cam, top])
        if vw is None:
            vw = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if i % 25 == 0:
            print(f"  {info}  assets={len(lt.assets())}")
    vw.release()
    cv2.imwrite(str(out / "demo_last_frame.png"), frame)
    twin = lt.finish(store_dir=out / "twin_store", dashboard_path=out / "Demo_Dashboard.html", session="live demo",
                     title=title, traj_gt=gt_traj)
    print("demo video:", out / "demo.mp4", "| dashboard:", out / "Demo_Dashboard.html", "|", lt.stats())
    return lt, twin
