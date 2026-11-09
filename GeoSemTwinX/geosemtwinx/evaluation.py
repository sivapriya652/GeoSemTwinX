"""Evaluation of semantic mapping / persistent identity against ground truth."""
from collections import Counter, defaultdict

import numpy as np

from .gasf import vkeys


def object_gt(segs):
    w = defaultdict(float)
    for s in segs:
        w[s.gt] += len(s.pts)
    g = max(w, key=w.get)
    purity = w[g] / sum(w.values())
    return g, purity


def evaluate_objects(objects, all_segs, instance_class, visible_gt):
    """objects: list of (label, segs, _). visible_gt: GT ids that were detectable.

    Returns precision/recall/F1 at asset level, duplicate rate, label accuracy,
    identity consistency (share of GT-labelled observations that end up in the
    single primary object of their true asset) and ID switches."""
    rows = []
    for k, (label, segs, _) in enumerate(objects):
        g, pur = object_gt(segs)
        rows.append(dict(k=k, label=label, gt=g, purity=pur, support=sum(len(s.pts) for s in segs), segs=segs))
    by_gt = defaultdict(list)
    for r in rows:
        if r["gt"] != 0 and r["purity"] >= 0.5:
            by_gt[r["gt"]].append(r)
    primary = {g: max(rs, key=lambda r: r["support"]) for g, rs in by_gt.items()}
    tp = len(primary)
    dup = sum(len(rs) - 1 for rs in by_gt.values())
    fp = len(rows) - tp - dup
    n_gt = len(visible_gt)
    prec = tp / max(len(rows), 1)
    rec = len(set(primary) & set(visible_gt)) / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    lab_acc = np.mean([primary[g]["label"] == instance_class.get(g) for g in primary]) if primary else 0.0
    # identity consistency over observations
    seg_obj = {}
    for r in rows:
        for s in r["segs"]:
            seg_obj[s.sid] = r["k"]
    prim_k = {g: r["k"] for g, r in primary.items()}
    obs = [s for s in all_segs if s.gt != 0]
    good = sum(1 for s in obs if seg_obj.get(s.sid) is not None and prim_k.get(s.gt) == seg_obj[s.sid])
    idc = good / max(len(obs), 1)
    switches = 0
    per_gt = defaultdict(list)
    for s in sorted(obs, key=lambda s: s.frame):
        if s.sid in seg_obj:
            per_gt[s.gt].append(seg_obj[s.sid])
    for seq in per_gt.values():
        switches += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return dict(objects=len(rows), tp=tp, duplicates=dup, false_pos=fp, gt_assets=n_gt,
                precision=prec, recall=rec, f1=f1, duplicate_rate=dup / max(tp, 1),
                label_acc=float(lab_acc), identity_consistency=idc, id_switches=switches,
                mean_purity=float(np.mean([r["purity"] for r in primary.values()])) if primary else 0.0)


def _enc(P, res):
    ijk = np.floor(P / res).astype(np.int64) + (1 << 20)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def gt_voxel_labels(seq, frame_ids, instance_class, classes, res=0.1, stride=4, max_depth=8.0, include_ids=None):
    """Voxelise GT classes from GT poses (majority vote per voxel). Returns {voxel key: class} where class is
    an asset class, 'background', or 'ignore' (assets outside include_ids, e.g. not detectable)."""
    from .mapping import backproject
    names = list(classes) + ["background", "ignore"]
    cid = {c: k for k, c in enumerate(names)}
    lut_cache = {}
    K_all, C_all = [], []
    inc = set(include_ids) if include_ids is not None else None
    for i in frame_ids:
        f = seq.frame(i)
        P, z, (v, u) = backproject(f["depth"], seq.K, seq.gt_poses[i], stride=stride, max_depth=max_depth)
        ids = f["instance"][v, u]
        cl = np.full(len(ids), cid["background"], np.int64)
        for a in np.unique(ids):
            if a == 0:
                continue
            c = instance_class.get(int(a))
            if c is None:
                continue
            m = ids == a
            cl[m] = cid["ignore"] if (c not in cid or (inc is not None and int(a) not in inc)) else cid[c]
        K_all.append(_enc(P, res)); C_all.append(cl)
    keys = np.concatenate(K_all); cls = np.concatenate(C_all)
    up, cnt = np.unique(np.stack([keys, cls], 1), axis=0, return_counts=True)
    order = np.lexsort((-cnt, up[:, 0]))
    up = up[order]
    first = np.ones(len(up), bool); first[1:] = up[1:, 0] != up[:-1, 0]
    return {int(k): names[int(c)] for k, c in up[first]}


def semantic_miou(objects, gt_vox, classes, res=0.1, T_align=np.eye(4)):
    """3D IoU per class on voxels the GT observed; 'ignore' voxels are excluded."""
    pred = {}
    for label, segs, inst in objects:
        P = inst.pts if inst is not None else np.vstack([s.pts for s in segs])
        P = P @ T_align[:3, :3].T + T_align[:3, 3]
        for k in np.unique(_enc(P, res)):
            pred[int(k)] = label
    ious = {}
    for c in classes:
        g = {k for k, v in gt_vox.items() if v == c}
        p = {k for k, v in pred.items() if v == c and gt_vox.get(k, "ignore") != "ignore"}
        if not g and not p:
            continue
        ious[c] = len(g & p) / max(len(g | p), 1)
    return dict(per_class=ious, miou=float(np.mean(list(ious.values()))) if ious else 0.0)
