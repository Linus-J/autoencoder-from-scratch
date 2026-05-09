"""Autoencoder — Python / PyTorch implementation

Hyperparameters mirror C/config.h for a fair wall-clock timing comparison.
Optimizer: plain Adam (β1=0.9, β2=0.999, ε=1e-8).
Data:      MNIST CSV files shared with the C implementation.
Device:    CPU only, so the comparison is compute-equivalent.
"""

import csv
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Hyperparameters (mirrors C/config.h) ─────────────────────────────────────
INPUT_DIM      = 784
HIDDEN1        = 512
HIDDEN2        = 128
LATENT_DIM     = 16
LEARNING_RATE  = 1e-3
EPOCHS         = 1
BATCH_SIZE     = 1
NUM_TRAIN_IMGS = 5000
_HERE          = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV      = os.path.join(_HERE, "..", "C", "data", "mnist_train.csv")

DEVICE = torch.device("cpu")   # CPU only — fair comparison with C

# ── Data loading ──────────────────────────────────────────────────────────────
def load_mnist_csv(path, n):
    """Return a (n, 784) float32 tensor with pixels normalised to [0, 1]."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                  # skip header
        for i, row in enumerate(reader):
            if i >= n:
                break
            rows.append([float(v) / 255.0 for v in row[1:]])
    return torch.tensor(rows, dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────────────────────────
class AutoEncoder(nn.Module):
    """784 → 512 → 128 → 16 → 128 → 512 → 784  (all ReLU activations)."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),  nn.ReLU(),
            nn.Linear(HIDDEN1,   HIDDEN2),  nn.ReLU(),
            nn.Linear(HIDDEN2,   LATENT_DIM), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2,    HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,    INPUT_DIM), nn.ReLU(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Training ──────────────────────────────────────────────────────────────────
def run(num_train=NUM_TRAIN_IMGS):
    """Train the autoencoder and return wall-clock training time in seconds."""
    data = load_mnist_csv(TRAIN_CSV, num_train)
    n_batches = num_train // BATCH_SIZE

    model     = AutoEncoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()

    model.train()
    wall_start = time.perf_counter()

    losses = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        t0 = time.perf_counter()
        for i in range(0, num_train, BATCH_SIZE):
            x = data[i:i + BATCH_SIZE].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            l = loss.item()
            epoch_loss += l
            losses.append(l)

        avg = epoch_loss / n_batches
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch+1}/{EPOCHS} — avg loss: {avg:.6f} — time: {elapsed:.2f} s")

    return time.perf_counter() - wall_start, losses

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Autoencoder — Python / PyTorch implementation ===")
    print(f"Architecture : {INPUT_DIM} → {HIDDEN1} → {HIDDEN2} → {LATENT_DIM}"
          f" → {HIDDEN2} → {HIDDEN1} → {INPUT_DIM}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training imgs: {NUM_TRAIN_IMGS}")
    print()
    t, _ = run()
    print()
    print("=== Training complete ===")
    print(f"Wall-clock time: {t:.3f} s")
