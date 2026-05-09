#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "config.h"
#include "util/img.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main(void) {
	omp_set_num_threads(AE_OMP_THREADS);
	srand((unsigned int)time(NULL));

	/* ── TRAINING ──────────────────────────────────────────────────────── */
	printf("=== Autoencoder — C implementation ===\n");
	printf("Architecture : %d → %d → %d → %d → %d → %d → %d\n",
	       AE_INPUT_DIM, AE_HIDDEN1, AE_HIDDEN2, AE_LATENT_DIM,
	       AE_HIDDEN2,   AE_HIDDEN1, AE_INPUT_DIM);
	printf("Epochs       : %d\n", AE_EPOCHS);
	printf("Batch size   : %d\n", AE_BATCH_SIZE);
	printf("Learning rate: %g\n", AE_LEARNING_RATE);
	printf("Training imgs: %d\n\n", AE_NUM_TRAIN_IMGS);

	Img **train_imgs = csv_to_imgs(TRAIN_CSV, AE_NUM_TRAIN_IMGS);
	NeuralNetwork *net = aeCreate(AE_LATENT_DIM, AE_LEARNING_RATE, AE_BATCH_SIZE);

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	network_train_batch_imgs(net, train_imgs, AE_NUM_TRAIN_IMGS,
	                         AE_BATCH_SIZE, AE_EPOCHS, AE_LATENT_DIM);

	clock_gettime(CLOCK_MONOTONIC, &t_end);
	double elapsed = (t_end.tv_sec  - t_start.tv_sec)
	               + (t_end.tv_nsec - t_start.tv_nsec) * 1e-9;
	printf("\n=== Training complete ===\n");
	printf("Wall-clock time: %.3f s\n\n", elapsed);

	network_save(net, SAVE_DIR);
	imgs_free(train_imgs, AE_NUM_TRAIN_IMGS);
	free(train_imgs);
	network_free(net);

	/* ── INFERENCE & VISUALISATION ─────────────────────────────────────── */
	printf("=== Inference ===\n");
	Img **test_imgs = csv_to_imgs(TEST_CSV, AE_NUM_TEST_IMGS);
	NeuralNetwork *loaded = network_load(SAVE_DIR);

	/* Save a strip of original test images (first AE_NUM_VIS_IMGS to match reconstructions) */
	img_save_new(test_imgs, AE_NUM_VIS_IMGS, "originalImages.pgm");

	/* Run encoder→decoder and save reconstructions */
	Img **out_imgs = malloc(AE_NUM_VIS_IMGS * sizeof(Img *));
	for (int i = 0; i < AE_NUM_VIS_IMGS; i++) {
		out_imgs[i] = network_predict(loaded, test_imgs[i]);
	}
	img_save_new(out_imgs, AE_NUM_VIS_IMGS, "compressedImages.pgm");

	printf("Saved original images  → originalImages.pgm\n");
	printf("Saved reconstructions  → compressedImages.pgm\n");

	imgs_free(test_imgs, AE_NUM_TEST_IMGS);
	free(test_imgs);
	imgs_free(out_imgs, AE_NUM_VIS_IMGS);
	free(out_imgs);
	network_free(loaded);

	return 0;
}
