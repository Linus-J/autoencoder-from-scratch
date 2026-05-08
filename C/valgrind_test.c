/*
 * valgrind_test.c — minimal pipeline test for memory leak checking.
 *
 * Trains on 50 images for 1 epoch, saves, loads, runs inference on 3 images,
 * then cleans everything up.  Run with:
 *
 *   make CFLAGS="-O0 -g"
 *   gcc -O0 -g -I. valgrind_test.c matrix/matrix.o matrix/ops.o \
 *       neural/activations.o neural/nn.o util/img.o util/ziggurat_inline.o \
 *       -o valgrind_test -lm
 *   valgrind --leak-check=full --error-exitcode=1 ./valgrind_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h"
#include "util/img.h"
#include "neural/nn.h"

#define VG_TRAIN_IMGS   50
#define VG_TEST_IMGS     3
#define VG_EPOCHS        1
#define VG_SAVE_DIR     "/tmp/vg_ae_test"

int main(void) {
    /* ── Training ───────────────────────────────────────────────────── */
    printf("[vg] Loading %d training images...\n", VG_TRAIN_IMGS);
    Img **train = csv_to_imgs(TRAIN_CSV, VG_TRAIN_IMGS);
    if (!train) { fprintf(stderr, "csv_to_imgs failed\n"); return 1; }

    NeuralNetwork *net = aeCreate(AE_LATENT_DIM, AE_LEARNING_RATE, AE_BATCH_SIZE);
    printf("[vg] Training...\n");
    network_train_batch_imgs(net, train, VG_TRAIN_IMGS,
                             AE_BATCH_SIZE, VG_EPOCHS, AE_LATENT_DIM);

    printf("[vg] Saving...\n");
    network_save(net, VG_SAVE_DIR);

    imgs_free(train, VG_TRAIN_IMGS);
    free(train);
    network_free(net);

    /* ── Inference ──────────────────────────────────────────────────── */
    printf("[vg] Loading network...\n");
    NeuralNetwork *loaded = network_load(VG_SAVE_DIR);
    if (!loaded) { fprintf(stderr, "network_load failed\n"); return 1; }

    printf("[vg] Loading %d test images...\n", VG_TEST_IMGS);
    Img **test = csv_to_imgs(TEST_CSV, VG_TEST_IMGS);
    if (!test) { fprintf(stderr, "csv_to_imgs(test) failed\n"); return 1; }

    for (int i = 0; i < VG_TEST_IMGS; i++) {
        Img *out = network_predict(loaded, test[i]);
        if (!out) { fprintf(stderr, "network_predict failed\n"); return 1; }
        img_free(out);
    }

    /* ── Cleanup ────────────────────────────────────────────────────── */
    imgs_free(test, VG_TEST_IMGS);
    free(test);
    network_free(loaded);

    printf("[vg] All done — check valgrind output for leaks.\n");
    return 0;
}
