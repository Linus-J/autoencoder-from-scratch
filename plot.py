#!/usr/bin/env python3
"""plot.py — run all three autoencoder implementations and generate comparison plots.

Usage:
    python plot.py

Outputs:
    plots/results.png  — convergence curves, wall-clock times, reconstructions
"""

import csv
import glob
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

_HERE  = os.path.dirname(os.path.abspath(__file__))
_PLOTS = os.path.join(_HERE, "plots")
os.makedirs(_PLOTS, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_module(label, path):
    """Load a Python source file or compiled extension as a named module."""
    spec = importlib.util.spec_from_file_location(label, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rolling_mean(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")

# ── 1. C ─────────────────────────────────────────────────────────────────────
print("── C ──────────────────────────────────────────────────────────────────")
c_dir = os.path.join(_HERE, "C")
subprocess.run(["make", "clean"], cwd=c_dir, check=False, capture_output=True)
subprocess.run(["make"], cwd=c_dir, check=True, capture_output=True)

result = subprocess.run(["./main"], cwd=c_dir, check=True,
                        capture_output=True, text=True)
print(result.stdout.strip())

c_wall = None
for line in result.stdout.splitlines():
    if "Wall-clock time:" in line:
        c_wall = float(line.split(":")[1].strip().rstrip(" s"))
        break

c_losses = []
with open(os.path.join(c_dir, "losses_c.csv")) as f:
    next(f)  # skip header
    for row in csv.reader(f):
        c_losses.append(float(row[1]))

# ── 2. Python / PyTorch ───────────────────────────────────────────────────────
print("\n── Python / PyTorch ───────────────────────────────────────────────────")
py_ae = load_module("py_ae", os.path.join(_HERE, "Python", "autoencoder.py"))
py_wall, py_losses = py_ae.run()

# ── 3. Cython / PyTorch ───────────────────────────────────────────────────────
print("\n── Cython / PyTorch ───────────────────────────────────────────────────")
cy_dir = os.path.join(_HERE, "Cython")
subprocess.run(
    [sys.executable, "setup.py", "build_ext", "--inplace", "-q"],
    cwd=cy_dir, check=True, capture_output=True,
)
so_files = glob.glob(os.path.join(cy_dir, "autoencoder.cpython-*.so"))
cy_ae    = load_module("autoencoder", so_files[0])
cy_wall, cy_losses = cy_ae.run()

# ── 4. Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel 1 — Convergence (top row, left two thirds) ----------------------------
ax_conv = fig.add_subplot(gs[0, :2])
W = 50
for losses, label, color in [
    (c_losses,  "C (from scratch)", "tab:blue"),
    (py_losses, "Python / PyTorch", "tab:orange"),
    (cy_losses, "Cython / PyTorch", "tab:green"),
]:
    s = rolling_mean(losses, W)
    ax_conv.plot(np.arange(len(s)), s, label=label, linewidth=1.5, color=color)

ax_conv.set_xlabel("Batch")
ax_conv.set_ylabel("MSE loss")
ax_conv.set_title(f"Convergence  (rolling mean, window = {W})")
ax_conv.legend()
ax_conv.grid(True, alpha=0.3)

# Panel 2 — Wall-clock bar chart (top right) ----------------------------------
ax_time = fig.add_subplot(gs[0, 2])
names  = ["C\n(from scratch)", "Python\n/ PyTorch", "Cython\n/ PyTorch"]
times  = [c_wall, py_wall, cy_wall]
colors = ["tab:blue", "tab:orange", "tab:green"]
bars   = ax_time.barh(names, times, color=colors, height=0.5)
ax_time.bar_label(bars, fmt="%.1f s", padding=5)
ax_time.set_xlabel("Wall-clock time (s)")
ax_time.set_title("Training time\n(5 000 images, 1 epoch)")
ax_time.invert_yaxis()
ax_time.set_xlim(0, max(times) * 1.28)
ax_time.grid(True, axis="x", alpha=0.3)

# Panel 3 — Reconstruction images (bottom row) --------------------------------
orig_path  = os.path.join(c_dir, "originalImages.pgm")
recon_path = os.path.join(c_dir, "compressedImages.pgm")
orig_img   = np.array(Image.open(orig_path))
recon_img  = np.array(Image.open(recon_path))

ax_orig  = fig.add_subplot(gs[1, :2])
ax_recon = fig.add_subplot(gs[1, 2])

ax_orig.imshow(orig_img,  cmap="gray", aspect="auto", vmin=0, vmax=255)
ax_orig.set_title("Original MNIST test images")
ax_orig.axis("off")

ax_recon.imshow(recon_img, cmap="gray", aspect="auto", vmin=0, vmax=255)
ax_recon.set_title("Reconstructions (C)")
ax_recon.axis("off")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out = os.path.join(_PLOTS, "results.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlots saved → {out}")
