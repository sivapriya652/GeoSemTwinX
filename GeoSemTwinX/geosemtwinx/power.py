"""GPU power / utilisation / memory sampling via nvidia-smi (works on Linux, WSL2 and Windows)."""
import shutil
import subprocess
import threading
import time

import numpy as np


def _smi():
    for c in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", r"C:\Windows\System32\nvidia-smi.exe"):
        if shutil.which(c) or (c.startswith("/") and __import__("os").path.exists(c)):
            return c
    return None


class GPUMonitor:
    """with GPUMonitor() as m: ...work...   ->  m.stats() = mean/peak power (W), utilisation (%), memory (MB), energy (J)"""

    def __init__(self, period=0.25):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.cmd = _smi()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run([self.cmd, "--query-gpu=power.draw,utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                p, u, m = [float(x) for x in out.strip().splitlines()[0].split(",")[:3]]
                self.samples.append((time.time(), p, u, m))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.t0 = time.time()
        if self.cmd:
            self.th = threading.Thread(target=self._run, daemon=True); self.th.start()
        return self

    def __exit__(self, *a):
        self.t1 = time.time(); self._stop.set()
        if self.cmd:
            self.th.join(timeout=6)

    def stats(self):
        dur = getattr(self, "t1", time.time()) - self.t0
        if not self.samples:
            return dict(available=False, seconds=dur)
        S = np.array(self.samples)
        return dict(available=True, seconds=dur, power_mean_w=float(S[:, 1].mean()), power_peak_w=float(S[:, 1].max()),
                    util_mean_pct=float(S[:, 2].mean()), mem_peak_mb=float(S[:, 3].max()),
                    energy_j=float(S[:, 1].mean() * dur))


def idle_power(seconds=3.0):
    with GPUMonitor() as m:
        time.sleep(seconds)
    return m.stats()
