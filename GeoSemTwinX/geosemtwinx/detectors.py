"""2D perception front ends.

GroundedSAM2Detector  - open-vocabulary detection (Grounding DINO) + promptable
                        segmentation (SAM 2.1). Needs a GPU runtime (Colab T4 is fine).
OracleNoisyDetector   - derives detections from ground-truth instance masks and
                        injects realistic detector errors (misses, label confusion,
                        mask bleed, split masks, false positives). Used to evaluate
                        the fusion stage in isolation with exact ground truth.
"""
import numpy as np
from scipy import ndimage


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


SAM2_CKPTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
}
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/{}"


def sam2_checkpoint(size="tiny", ckpt_dir="checkpoints"):
    """Download (once) and return (checkpoint path, config name) for a SAM 2.1 model size."""
    import os
    import urllib.request
    fn, cfg = SAM2_CKPTS[size]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, fn)
    if not os.path.exists(path):
        print("Downloading", fn, "...")
        urllib.request.urlretrieve(SAM2_URL.format(fn), path)
    return path, cfg


class GroundedSAM2Detector:
    """Grounding DINO (open-vocabulary boxes) + SAM 2.1 (masks). Runs on CUDA when available.

    dino: "IDEA-Research/grounding-dino-tiny" or "IDEA-Research/grounding-dino-base"
    sam_size: "tiny" | "small" | "base_plus"
    The per-call timings (ms) of the last frame are kept in self.last_timing."""

    def __init__(self, classes, device=None, dino_id="IDEA-Research/grounding-dino-tiny", sam_size="tiny",
                 sam_cfg=None, sam_ckpt=None, ckpt_dir="checkpoints", box_threshold=0.30, text_threshold=0.25,
                 max_box_frac=0.5, nms_iou=0.7, max_dets=30):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = list(classes)
        self.prompt = " ".join(c + "." for c in self.classes)
        self.proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(self.device).eval()
        if sam_ckpt is None:
            sam_ckpt, sam_cfg = sam2_checkpoint(sam_size, ckpt_dir)
        self.sam = SAM2ImagePredictor(build_sam2(sam_cfg, sam_ckpt, device=self.device))
        self.bt, self.tt, self.max_frac, self.nms, self.max_dets = box_threshold, text_threshold, max_box_frac, nms_iou, max_dets
        import inspect
        params = inspect.signature(self.proc.post_process_grounded_object_detection).parameters
        self._thr_kw = "threshold" if "threshold" in params else "box_threshold"     # transformers >=4.51 vs older
        self._has_text_labels = "text_labels" in params
        self.last_timing = {}

    def _cls(self, text):
        text = text.lower()
        for c in sorted(self.classes, key=len, reverse=True):     # longest match first
            if c in text or text in c:
                return c
        for c in self.classes:
            if any(w in text for w in c.split()):
                return c
        return None

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def __call__(self, rgb):
        import time
        from PIL import Image
        torch = self.torch
        H, W = rgb.shape[:2]
        t0 = time.perf_counter()
        inputs = self.proc(images=Image.fromarray(rgb), text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dino(**inputs)
        kw = {self._thr_kw: self.bt, "text_threshold": self.tt, "target_sizes": [(H, W)]}
        if self._has_text_labels:
            kw["text_labels"] = [self.classes]
        res = self.proc.post_process_grounded_object_detection(out, input_ids=inputs["input_ids"], **kw)[0]
        labels = res.get("text_labels", res.get("labels"))
        dets = []
        for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            c = self._cls(str(lab))
            if c is None or (box[2] - box[0]) * (box[3] - box[1]) > self.max_frac * W * H:
                continue
            dets.append(dict(label=c, score=float(score), box=box))
        dets.sort(key=lambda d: -d["score"])
        kept = []
        for d in dets:
            if all(box_iou(d["box"], k["box"]) < self.nms for k in kept):
                kept.append(d)
        kept = kept[: self.max_dets]
        self._sync(); t1 = time.perf_counter()
        if kept:
            dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode(), torch.autocast(self.device, dtype=dtype, enabled=self.device == "cuda"):
                self.sam.set_image(rgb)
                for k in kept:
                    masks, mscores, _ = self.sam.predict(box=k["box"].astype(np.float32), multimask_output=False)
                    k["mask"] = np.asarray(masks[0]).astype(bool)
                    k["mask_score"] = float(np.asarray(mscores, dtype=np.float32).ravel()[0])
        self._sync(); t2 = time.perf_counter()
        self.last_timing = dict(dino_ms=1000 * (t1 - t0), sam_ms=1000 * (t2 - t1), n=len(kept))
        return kept


class OracleNoisyDetector:
    """Ground-truth masks + error model calibrated to typical open-vocabulary detector behaviour."""

    def __init__(self, classes, instance_class, seed=0, p_miss=0.15, p_miss_small=0.6, min_area=350,
                 p_confuse=0.08, p_bleed=0.15, p_split=0.06, p_dup=0.05, p_fp=0.12, confusions=None):
        self.classes = list(classes)
        self.instance_class = instance_class
        self.rng = np.random.default_rng(seed)
        self.p = dict(miss=p_miss, miss_small=p_miss_small, confuse=p_confuse, bleed=p_bleed,
                      split=p_split, dup=p_dup, fp=p_fp)
        self.min_area = min_area
        self.conf = confusions or {"pallet": "cardboard box", "cardboard box": "pallet",
                                   "barrel": "cardboard box", "forklift": "pallet truck"}

    def __call__(self, frame):
        rng, p = self.rng, self.p
        inst = frame["instance"]
        H, W = inst.shape
        dets = []
        ids, cnt = np.unique(inst, return_counts=True)
        for iid, area in zip(ids, cnt):
            if iid == 0 or iid not in self.instance_class:
                continue
            cls = self.instance_class[iid]
            if cls not in self.classes:
                continue
            if rng.random() < (p["miss_small"] if area < self.min_area else p["miss"]):
                continue
            m = inst == iid
            label = cls
            if rng.random() < p["confuse"]:
                alt = self.conf.get(cls)
                if alt in self.classes:
                    label = alt
            r = rng.random()
            if r < p["bleed"]:
                m = ndimage.binary_dilation(m, iterations=int(rng.integers(6, 14)))
            elif r < p["bleed"] + 0.3:
                it = int(rng.integers(1, 3))
                m = ndimage.binary_erosion(m, iterations=it) if rng.random() < 0.5 else ndimage.binary_dilation(m, iterations=it)
            score = float(np.clip(rng.normal(0.48, 0.07), 0.3, 0.8))
            parts = [m]
            if rng.random() < p["split"] and area > 2000:
                ys, xs = np.nonzero(m)
                cut = np.median(xs)
                a = m.copy(); a[:, int(cut):] = False
                b = m.copy(); b[:, :int(cut)] = False
                parts = [a, b]
            for part in parts:
                if part.sum() < 50:
                    continue
                dets.append(dict(label=label, score=score, mask=part, mask_score=float(rng.uniform(0.85, 0.98))))
            if rng.random() < p["dup"]:
                dets.append(dict(label=label, score=score * 0.8, mask=ndimage.binary_erosion(m, iterations=2),
                                 mask_score=0.9))
        if rng.random() < p["fp"]:                     # hallucinated object on background
            cy, cx = int(rng.integers(H // 4, 3 * H // 4)), int(rng.integers(W // 6, 5 * W // 6))
            hh, ww = int(rng.integers(20, 70)), int(rng.integers(20, 90))
            m = np.zeros((H, W), bool); m[max(cy - hh, 0):cy + hh, max(cx - ww, 0):cx + ww] = True
            m &= frame["depth"] > 0
            if m.sum() > 200:
                dets.append(dict(label=self.classes[int(rng.integers(len(self.classes)))],
                                 score=float(rng.uniform(0.3, 0.42)), mask=m, mask_score=float(rng.uniform(0.6, 0.85))))
        return dets
