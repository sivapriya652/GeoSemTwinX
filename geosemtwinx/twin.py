"""Persistent digital twin: versioned asset registry, spatial queries, store, 3D viewer."""
import datetime as _dt
import json
import os
from pathlib import Path

import numpy as np


def _r(x, n=3):
    return [round(float(v), n) for v in np.asarray(x).ravel()]


class DigitalTwin:
    def __init__(self, name="GeoSemTwin-X"):
        self.name = name
        self.version = 0
        self.assets = {}          # id -> record
        self.history = []         # list of version summaries
        self.changes = []
        self._pts = {}            # asset id -> world points (kept in memory for change detection)
        self._occ = None

    # ------------------------------------------------------------ build / update
    def commit(self, instances, classes, session, T_align=np.eye(4), changes=None, map_stats=None):
        """instances: confirmed GASF Instance objects (estimated world frame)."""
        self.version += 1
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        seen = set()
        for inst in instances:
            P = inst.pts @ T_align[:3, :3].T + T_align[:3, 3]
            aid = getattr(inst, "asset_id", None) or f"A{len(self.assets) + 1:04d}"
            inst.asset_id = aid
            lo, hi = np.percentile(P, [2, 98], axis=0)           # robust box: ignore stray points
            core = P[np.all((P >= lo) & (P <= hi), axis=1)]
            P = core if len(core) >= 10 else P
            rec = self.assets.get(aid, dict(id=aid, first_seen=ts, versions=[]))
            rec.update(label=classes[int(np.argmax(inst.probs))], confidence=round(float(inst.probs.max()), 3),
                       label_entropy=round(inst.label_entropy(), 3), centroid=_r(np.median(P, 0)),
                       bbox_min=_r(lo), bbox_max=_r(hi), size=_r(hi - lo), observations=int(inst.hits),
                       views=len(inst.views), last_seen=ts, status="present", session=session)
            rec["versions"].append(dict(version=self.version, centroid=rec["centroid"], size=rec["size"]))
            self.assets[aid] = rec
            self._pts[aid] = P
            seen.add(aid)
        for c in changes or []:
            c = dict(c, version=self.version, time=ts)
            self.changes.append(c)
            if c["type"] == "removed" and c.get("a") in self.assets:
                self.assets[c["a"]]["status"] = "removed"
        self.history.append(dict(version=self.version, time=ts, session=session, n_assets=len(seen),
                                 map=map_stats or {}))
        return self.version

    def update(self, instances, classes, session, occ_prev=None, occ_new=None, T_align=np.eye(4)):
        """Commit a new mapping session against the current twin: assets keep their IDs when re-observed
        (moved ones are re-positioned), new ones get new IDs, removals are verified with occupancy."""
        from .change import detect_changes
        if self.version == 0:
            v = self.commit(instances, classes, session, T_align)
            self._occ = occ_new
            return v, []
        prev = [dict(id=a["id"], label=a["label"], centroid=np.array(a["centroid"]), extent=np.array(a["size"]),
                     pts=self._pts[a["id"]]) for a in self.present() if a["id"] in self._pts]
        new = []
        for k, inst in enumerate(instances):
            P = inst.pts @ T_align[:3, :3].T + T_align[:3, 3]
            new.append(dict(id=f"new{k}", label=classes[int(np.argmax(inst.probs))], centroid=P.mean(0),
                            extent=np.ptp(P, 0), pts=P, inst=inst))
        ch = detect_changes(prev, new, occ_prev if occ_prev is not None else getattr(self, "_occ", None), occ_new)
        keep = {}
        for c in ch:
            if c["type"] in ("unchanged", "moved", "geometry"):
                keep[c["b"]] = c["a"]
        for n in new:
            n["inst"].asset_id = keep.get(n["id"])            # None -> commit() assigns a fresh ID
        log = []
        for c in ch:
            if c["type"] == "unchanged":
                continue
            e = dict(c)
            if "b" in e:
                e["b"] = keep.get(e["b"], e["b"])
            log.append(e)
        v = self.commit(instances, classes, session, T_align, changes=[])
        id_of = {n["id"]: n["inst"].asset_id for n in new}
        for e in log:
            if e["type"] in ("added", "unverified") and "b" in e:
                e["b"] = id_of.get(e["b"], e["b"])
            e.update(version=self.version, time=self.history[-1]["time"])
            self.changes.append(e)
            if e["type"] == "removed" and e.get("a") in self.assets:
                self.assets[e["a"]]["status"] = "removed"
        self._occ = occ_new
        return v, log

    # ------------------------------------------------------------ spatial queries
    def present(self):
        return [a for a in self.assets.values() if a["status"] == "present"]

    def query_radius(self, point, radius, label=None):
        p = np.asarray(point, float)
        out = []
        for a in self.present():
            if label and a["label"] != label:
                continue
            d = float(np.linalg.norm(np.array(a["centroid"]) - p))
            if d <= radius:
                out.append((d, a))
        return [a for _, a in sorted(out, key=lambda x: x[0])]

    def query_box(self, lo, hi, label=None):
        lo, hi = np.asarray(lo), np.asarray(hi)
        return [a for a in self.present() if (label is None or a["label"] == label)
                and np.all(np.array(a["centroid"]) >= lo) and np.all(np.array(a["centroid"]) <= hi)]

    def nearest(self, point, label):
        c = [a for a in self.present() if a["label"] == label]
        if not c:
            return None
        return min(c, key=lambda a: np.linalg.norm(np.array(a["centroid"]) - np.asarray(point)))

    def counts(self):
        out = {}
        for a in self.present():
            out[a["label"]] = out.get(a["label"], 0) + 1
        return out

    # ------------------------------------------------------------ persistence
    def to_dict(self):
        return dict(name=self.name, version=self.version, assets=list(self.assets.values()),
                    history=self.history, changes=self.changes)


class TwinStore:
    """Versioned store. Point `root` at a synced folder (e.g. a mounted Google Drive
    path in Colab, or an S3/GCS fuse mount) to get cloud synchronisation."""

    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def push(self, twin):
        d = twin.to_dict()
        p = self.root / f"twin_v{twin.version:03d}.json"
        p.write_text(json.dumps(d, indent=1))
        idx = self.root / "index.json"
        index = json.loads(idx.read_text()) if idx.exists() else []
        index.append(dict(version=twin.version, file=p.name, n_assets=len(twin.present()),
                          time=twin.history[-1]["time"]))
        idx.write_text(json.dumps(index, indent=1))
        return p

    def versions(self):
        idx = self.root / "index.json"
        return json.loads(idx.read_text()) if idx.exists() else []

    def load(self, version):
        return json.loads((self.root / f"twin_v{version:03d}.json").read_text())


# ---------------------------------------------------------------- viewer
CLASS_COLORS = {"pallet": "#b07a3c", "cardboard box": "#d9a35b", "forklift": "#f2b705", "barrel": "#2e8b57",
                "rack": "#3366cc", "monitor": "#555", "keyboard": "#888", "cup": "#c33", "chair": "#836",
                "book": "#365", "mouse": "#999", "laptop": "#444"}


def _box_lines(lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    X, Y, Z = [], [], []
    for a, b in e:
        X += [c[a][0], c[b][0], None]; Y += [c[a][1], c[b][1], None]; Z += [c[a][2], c[b][2], None]
    return X, Y, Z


def viewer_html(path, twin, map_points=None, map_colors=None, traj_est=None, traj_gt=None, title=None,
                max_points=60000):
    import plotly.graph_objects as go
    fig = go.Figure()
    if map_points is not None and len(map_points):
        idx = np.random.default_rng(0).choice(len(map_points), min(max_points, len(map_points)), replace=False)
        P = map_points[idx]
        col = None
        if map_colors is not None:
            Cc = (np.asarray(map_colors)[idx] * 255).astype(int)
            col = [f"rgb({r},{g},{b})" for r, g, b in Cc]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="map",
                                   marker=dict(size=1.2, color=col or "#9aa", opacity=0.6), hoverinfo="skip"))
    for a in twin.present():
        X, Y, Z = _box_lines(a["bbox_min"], a["bbox_max"])
        col = CLASS_COLORS.get(a["label"], "#e4572e")
        fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z, mode="lines", line=dict(color=col, width=5),
                                   name=f'{a["id"]} {a["label"]}', legendgroup=a["label"], showlegend=False,
                                   hoverinfo="skip"))
        c = a["centroid"]
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[a["bbox_max"][2] + 0.15], mode="markers+text", text=[a["id"]],
            textposition="top center", marker=dict(size=4, color=col), name=a["label"], legendgroup=a["label"],
            showlegend=False,
            hovertext=[f'{a["id"]} | {a["label"]} ({a["confidence"]:.2f})<br>size {a["size"]} m<br>'
                       f'obs {a["observations"]}, views {a["views"]}'], hoverinfo="text"))
    for traj, name, color in ((traj_gt, "ground-truth path", "#2a9d8f"), (traj_est, "estimated path", "#e76f51")):
        if traj is not None and len(traj):
            T = np.asarray(traj)
            fig.add_trace(go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines", name=name,
                                       line=dict(color=color, width=4)))
    for ch in twin.changes:
        if ch["type"] in ("moved", "added", "removed") and ("b" in ch or "a" in ch):
            aid = ch.get("b") or ch.get("a")
            if aid in twin.assets:
                c = twin.assets[aid]["centroid"]
                sym = {"moved": "diamond", "added": "circle", "removed": "x"}[ch["type"]]
                fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[1]], z=[c[2] + 1.2], mode="markers+text",
                                           text=[ch["type"]], marker=dict(size=7, symbol=sym, color="#d62828"),
                                           showlegend=False))
    fig.update_layout(title=title or f"{twin.name} v{twin.version}", scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing="constant"))
    fig.write_html(path, include_plotlyjs="cdn")
    return path
