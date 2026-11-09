"""Structural / asset change detection between two digital-twin versions."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def chamfer(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf
    da, _ = cKDTree(B).query(A); db, _ = cKDTree(A).query(B)
    return float(0.5 * (np.median(da) + np.median(db)))


def detect_changes(assets_a, assets_b, occ_a=None, occ_b=None, move_thr=0.5, geom_thr=0.25, gate=4.0):
    """assets_*: list of dict(id, label, centroid, extent, pts).
    occ_*: OccupancyMap of each session. A change is only reported when the other session actually
    observed the relevant space: a move needs the old spot to be free in B (and the new spot free in A),
    a removal needs A's spot free in B, an addition needs B's spot free in A. Otherwise -> 'unverified'."""
    na, nb = len(assets_a), len(assets_b)
    C = np.full((na, nb), 1e3)
    for i, a in enumerate(assets_a):
        for j, b in enumerate(assets_b):
            if a["label"] != b["label"]:
                continue
            d = np.linalg.norm(a["centroid"] - b["centroid"])
            if d > gate:
                continue
            size = np.abs(np.log((a["extent"] + 0.05) / (b["extent"] + 0.05))).sum()
            C[i, j] = d + 0.5 * size
    rows, cols = linear_sum_assignment(C) if na and nb else ([], [])
    pairs = [(r, c) for r, c in zip(rows, cols) if C[r, c] < 1e3]
    out, ma, mb = [], set(), set()
    for r, c in pairs:
        a, b = assets_a[r], assets_b[c]
        d = float(np.linalg.norm(a["centroid"] - b["centroid"]))
        thr = max(move_thr, 0.4 * float(max(a["extent"].max(), b["extent"].max())))
        if d < thr:
            ma.add(r); mb.add(c)                                    # same asset, same place
            out.append(dict(type="unchanged", a=a["id"], b=b["id"], label=a["label"], displacement=d))
            ch = chamfer(a["pts"] - a["centroid"], b["pts"] - b["centroid"])
            ext = np.abs(a["extent"] - b["extent"]).max()
            if ch > geom_thr and ext > geom_thr:
                out.append(dict(type="geometry", a=a["id"], b=b["id"], label=a["label"], chamfer=ch))
        elif _observed_free(occ_b, a["pts"]) and _observed_free(occ_a, b["pts"]):
            ma.add(r); mb.add(c)                                    # verified move
            out.append(dict(type="moved", a=a["id"], b=b["id"], label=a["label"], displacement=d))
        # else: leave both unmatched; handled below as removed / added / unverified
    for i, a in enumerate(assets_a):
        if i in ma:
            continue
        if occ_b is None or _observed_free(occ_b, a["pts"]):
            out.append(dict(type="removed", a=a["id"], label=a["label"]))
        elif not _observed(occ_b, a["pts"]):
            out.append(dict(type="unverified", a=a["id"], label=a["label"]))
    for j, b in enumerate(assets_b):
        if j in mb:
            continue
        if occ_a is None or _observed_free(occ_a, b["pts"]):
            out.append(dict(type="added", b=b["id"], label=b["label"]))
        elif not _observed(occ_a, b["pts"]):
            out.append(dict(type="unverified", b=b["id"], label=b["label"]))
    return out


def _observed(occ, pts, min_frac=0.5):
    ijk, ok = occ.index(pts)
    if not ok.any():
        return False
    return (occ.state()[occ.flat(ijk[ok])] >= 0).mean() >= min_frac


def _observed_free(occ, pts, min_frac=0.5):
    """True if the other session observed this space and found it free (not occupied)."""
    if occ is None:
        return True
    ijk, ok = occ.index(pts)
    if not ok.any():
        return False
    st = occ.state()[occ.flat(ijk[ok])]
    observed = st >= 0
    return observed.mean() >= min_frac and (st[observed] == 0).mean() >= 0.5


def score_changes(pred, gt_changes, name_a, name_b):
    """pred items reference asset ids; name_* map asset id -> GT object name (or None)."""
    gt = {(c["name"], c["type"]) for c in gt_changes}
    P = set()
    for p in pred:
        if p["type"] in ("unchanged", "unverified"):
            continue
        if p["type"] in ("moved", "geometry"):
            n = name_a.get(p["a"]) or name_b.get(p["b"])
            P.add((n, "moved" if p["type"] == "moved" else "geometry"))
        elif p["type"] == "removed":
            P.add((name_a.get(p["a"]), "removed"))
        elif p["type"] == "added":
            P.add((name_b.get(p["b"]), "added"))
    tp = len(P & gt)
    prec = tp / max(len(P), 1); rec = tp / max(len(gt), 1)
    return dict(tp=tp, predicted=len(P), gt=len(gt), precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9), pred=sorted(P, key=str), gt_list=sorted(gt))
