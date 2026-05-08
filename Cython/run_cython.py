"""Runner for the compiled Cython autoencoder module.

Build first:  python setup.py build_ext --inplace
Then run:     python run_cython.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import autoencoder   # the compiled .so

print("=== Autoencoder — Cython / PyTorch implementation ===")
print(f"Architecture : {autoencoder.INPUT_DIM} → {autoencoder.HIDDEN1}"
      f" → {autoencoder.HIDDEN2} → {autoencoder.LATENT_DIM}"
      f" → {autoencoder.HIDDEN2} → {autoencoder.HIDDEN1} → {autoencoder.INPUT_DIM}")
print(f"Epochs       : {autoencoder.EPOCHS}")
print(f"Batch size   : {autoencoder.BATCH_SIZE}")
print(f"Learning rate: {autoencoder.LEARNING_RATE}")
print(f"Training imgs: {autoencoder.NUM_TRAIN_IMGS}")
print()

t = autoencoder.run()

print()
print("=== Training complete ===")
print(f"Wall-clock time: {t:.3f} s")
