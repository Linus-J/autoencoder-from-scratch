#!/usr/bin/env python3
"""plot.py — run all three autoencoder implementations and generate comparison plots.

Usage:
    python plot.py [--epochs N]

    --epochs N   Number of training epochs (default: 1).
                 Use 10–100 for better-looking reconstructions.

Outputs:
    plots/results.png
"""

import argparse
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
import torch

_HERE  = os.path.dirname(os.path.abspath(__file__))
_PLOTS = os.path.join(_HERE, "plots")
os.makedirs(_PLOTS, exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs (default: 1)")
args   = parser.parse_args()
EPOCHS = args.epochs

N_VIS = 5   # number of test images to show as reconstructions
# Indices into mnist_test.csv (0-based, after header).
# Labels: 7, 1, 0, 4, 6 — varied and well-reconstructed digits.
VIS_INDICES = [0, 2, 3, 4, 11]

# ── helpers ───────────────────────────────────────────────────────────────────
def load_module(label, path):
    spec = importlib.util.spec_from_file_location(label, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def rolling_mean(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")

def load_test_images(indices):
    """Load specific rows from mnist_test.csv; return (n, 28, 28) float32 in [0, 1]."""
    path    = os.path.join(_HERE, "C", "data", "mnist_test.csv")
    idx_set = set(indices)
    imgs    = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i in idx_set:
                imgs[i] = np.array([float(v) / 255.0 for v in row[1:]],
                                   dtype=np.float32).reshape(28, 28)
            if len(imgs) == len(indices):
                break
    return np.stack([imgs[i] for i in indices])  # preserves requested order

def split_pgm_strip(path, indices):
    """Split a PGM strip produced by the C program into individual images.

    The C program writes images 0..max(indices) sequentially; we pick out
    only the columns corresponding to the requested indices.
    """
    img    = np.array(Image.open(path)).astype(np.float32) / 255.0
    n_cols = img.shape[1] // 28
    frames = [img[:, c * 28:(c + 1) * 28] for c in range(n_cols)]
    # The C program writes test images 0..max(indices) sequentially; pick the right columns.
    return np.stack([frames[k] for k in indices])

# ── Load shared test images ───────────────────────────────────────────────────
originals   = load_test_images(VIS_INDICES)                          # (5, 28, 28)
test_tensor = torch.tensor(originals.reshape(N_VIS, -1),
                           dtype=torch.float32)                      # (5, 784)

# ── 1. C ─────────────────────────────────────────────────────────────────────
print("── C ──────────────────────────────────────────────────────────────────")
c_dir = os.path.join(_HERE, "C")
subprocess.run(["make", "clean"], cwd=c_dir, check=False, capture_output=True)
subprocess.run(["make"], cwd=c_dir, check=True, capture_output=True)

vis_count = max(VIS_INDICES) + 1   # C writes images 0..vis_count-1 sequentially

result = subprocess.run(["./main", "--epochs", str(EPOCHS),
                         "--vis-count", str(vis_count)], cwd=c_dir,
                        check=True, stdout=subprocess.PIPE, stderr=None,
                        text=True)
print(result.stdout.strip())

c_wall = None
for line in result.stdout.splitlines():
    if "Wall-clock time:" in line:
        c_wall = float(line.split(":")[1].strip().rstrip(" s"))
        break

c_losses = []
with open(os.path.join(c_dir, "losses_c.csv")) as f:
    next(f)
    for row in csv.reader(f):
        c_losses.append(float(row[1]))

c_recons = split_pgm_strip(os.path.join(c_dir, "compressedImages.pgm"), VIS_INDICES)

# ── 2. Python / PyTorch ───────────────────────────────────────────────────────
print("\n── Python / PyTorch ───────────────────────────────────────────────────")
py_ae = load_module("py_ae", os.path.join(_HERE, "Python", "autoencoder.py"))
py_wall, py_losses, py_recons = py_ae.run(num_epochs=EPOCHS, test_x=test_tensor)

# ── 3. Cython / PyTorch ───────────────────────────────────────────────────────
print("\n── Cython / PyTorch ───────────────────────────────────────────────────")
cy_dir   = os.path.join(_HERE, "Cython")
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace", "-q"],
               cwd=cy_dir, check=True, capture_output=True)
so_files = glob.glob(os.path.join(cy_dir, "autoencoder.cpython-*.so"))
cy_ae    = load_module("autoencoder", so_files[0])
cy_wall, cy_losses, cy_recons = cy_ae.run(num_epochs=EPOCHS, test_x=test_tensor)

# ── 4. Figure ─────────────────────────────────────────────────────────────────
epoch_label = f"{EPOCHS} epoch{'s' if EPOCHS != 1 else ''}"

fig    = plt.figure(figsize=(14, 11))
outer  = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[1, 1.15], hspace=0.42)

# Top half — convergence curve + timing bar chart ─────────────────────────────
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                       width_ratios=[2, 1], wspace=0.35)

ax_conv = fig.add_subplot(top[0])
W = max(20, len(c_losses) // 100)
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

ax_time = fig.add_subplot(top[1])
names  = ["C\n(from scratch)", "Python\n/ PyTorch", "Cython\n/ PyTorch"]
times  = [c_wall, py_wall, cy_wall]
colors = ["tab:blue", "tab:orange", "tab:green"]
bars   = ax_time.barh(names, times, color=colors, height=0.5)
ax_time.bar_label(bars, fmt="%.1f s", padding=5)
ax_time.set_xlabel("Wall-clock time (s)")
ax_time.set_title(f"Training time\n(5 000 images, {epoch_label})")
ax_time.invert_yaxis()
ax_time.set_xlim(0, max(times) * 1.28)
ax_time.grid(True, axis="x", alpha=0.3)

# Bottom half — 4 rows × N_VIS cols of 28×28 images ──────────────────────────
img_rows = [
    ("Original", originals),
    ("C",        c_recons),
    ("Python",   py_recons),
    ("Cython",   cy_recons),
]

bottom = gridspec.GridSpecFromSubplotSpec(
    len(img_rows), N_VIS, subplot_spec=outer[1],
    hspace=0.06, wspace=0.04)

for r, (label, imgs) in enumerate(img_rows):
    for c in range(N_VIS):
        ax = fig.add_subplot(bottom[r, c])
        ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if c == 0:
            ax.set_ylabel(label, fontsize=9, labelpad=6,
                          rotation=0, ha="right", va="center")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out = os.path.join(_PLOTS, "results.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlots saved → {out}")

