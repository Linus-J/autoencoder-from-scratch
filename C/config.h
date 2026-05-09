#pragma once

/*
 * config.h — Autoencoder configuration
 *
 * Edit the values in this file to customise the network architecture and
 * training hyperparameters without touching any other source file.
 *
 * Architecture (matches Python/autoencoder.py):
 *   INPUT → HIDDEN1 → HIDDEN2 → LATENT → HIDDEN2 → HIDDEN1 → INPUT
 *   784   →  512   →   128   →   16   →   128   →   512   →  784
 */

/* ── Network architecture ──────────────────────────────────────────────── */

#define AE_INPUT_DIM   784   /* MNIST pixel count (28 × 28)                 */
#define AE_HIDDEN1     512   /* Encoder layer-1 / decoder layer-3 size      */
#define AE_HIDDEN2     128   /* Encoder layer-2 / decoder layer-2 size      */
#define AE_LATENT_DIM   16   /* Bottleneck (latent space) dimension          */

/* ── Training hyperparameters ──────────────────────────────────────────── */

#define AE_LEARNING_RATE  1e-3  /* Adam learning rate                        */
#define AE_EPOCHS            1  /* Number of full passes over the dataset    */
#define AE_BATCH_SIZE        1  /* Mini-batch size (keep at 1 to match Python)*/
#define AE_NUM_TRAIN_IMGS 5000  /* Images drawn from the training CSV        */
#define AE_NUM_TEST_IMGS    10  /* Images loaded for inference/visualisation */
#define AE_NUM_VIS_IMGS      5  /* Reconstructed images saved to disk        */

/* ── Parallelism ───────────────────────────────────────────────────────── */

/* Number of OpenMP threads used for Adam weight updates.
 * Scales best at ~8 threads for batch_size=1; reduce if you see slowdowns.
 * Override at runtime with the OMP_NUM_THREADS environment variable. */
#define AE_OMP_THREADS   8

/* ── Adam optimiser parameters ─────────────────────────────────────────── */

#define ADAM_BETA1   0.9    /* First-moment (momentum) decay rate            */
#define ADAM_BETA2   0.999  /* Second-moment (RMSprop) decay rate            */
#define ADAM_EPS     1e-8   /* Numerical-stability epsilon                   */

/* ── File paths ─────────────────────────────────────────────────────────── */

#define TRAIN_CSV  "data/mnist_train.csv"
#define TEST_CSV   "data/mnist_test.csv"
#define SAVE_DIR   "testing_net"
