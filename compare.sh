#!/usr/bin/env bash
# compare.sh — build and time all three autoencoder implementations
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv/bin/python"

separator() { printf '\n'; printf '%.0s━' {1..56}; printf '\n %s\n' "$1"; printf '%.0s━' {1..56}; printf '\n\n'; }

# ── C ─────────────────────────────────────────────────────────────────────────
separator "C (from scratch)"
(cd "$REPO/C" && make clean -q 2>/dev/null || true; make -q && ./main)

# ── Python / PyTorch ──────────────────────────────────────────────────────────
separator "Python / PyTorch"
"$VENV" "$REPO/Python/autoencoder.py"

# ── Cython / PyTorch ──────────────────────────────────────────────────────────
separator "Cython / PyTorch"
(cd "$REPO/Cython" && "$VENV" setup.py build_ext --inplace -q 2>/dev/null && "$VENV" run_cython.py)
