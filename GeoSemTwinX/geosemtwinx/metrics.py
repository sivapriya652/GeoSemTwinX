"""Trajectory metrics (TUM-benchmark style)."""
import numpy as np


def umeyama(src, dst, with_scale=False):
    """Find s,R,t minimising ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(est, gt, with_scale=False):
    """Absolute trajectory error after SE(3)/Sim(3) alignment. Returns dict + aligned positions."""
    pe = np.array([T[:3, 3] for T in est]); pg = np.array([T[:3, 3] for T in gt])
    s, R, t = umeyama(pe, pg, with_scale)
    al = (s * (R @ pe.T)).T + t
    err = np.linalg.norm(al - pg, axis=1)
    T_align = np.eye(4); T_align[:3, :3] = s * R; T_align[:3, 3] = t
    return dict(rmse=float(np.sqrt((err ** 2).mean())), mean=float(err.mean()),
                median=float(np.median(err)), max=float(err.max()), T_align=T_align,
                errors=err, aligned=al, gt=pg)


def rpe(est, gt, timestamps, delta_s=1.0):
    """Relative pose error over a fixed time interval (default 1 s)."""
    ts = np.asarray(timestamps)
    terr, rerr = [], []
    for i in range(len(est)):
        j = int(np.searchsorted(ts, ts[i] + delta_s))
        if j >= len(est):
            break
        E = np.linalg.inv(np.linalg.inv(gt[i]) @ gt[j]) @ (np.linalg.inv(est[i]) @ est[j])
        terr.append(np.linalg.norm(E[:3, 3]))
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
    terr, rerr = np.array(terr), np.array(rerr)
    return dict(trans_rmse=float(np.sqrt((terr ** 2).mean())), rot_rmse_deg=float(np.sqrt((rerr ** 2).mean())),
                n=len(terr))


def path_length(poses):
    p = np.array([T[:3, 3] for T in poses])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
