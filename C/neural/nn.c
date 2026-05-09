#include "nn.h"
#include "../config.h"
#include <sys/stat.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../util/ziggurat_inline.h"

#define MAXCHAR 1000

/* Architecture: AE_INPUT_DIM → AE_HIDDEN1 → AE_HIDDEN2 → latentDim
 *                            → AE_HIDDEN2 → AE_HIDDEN1 → AE_INPUT_DIM  */
NeuralNetwork* aeCreate(int latentDim, double lr, int batchSize) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->learning_rate = lr;

	/* Encoder */
	net->X         = AE_INPUT_DIM;
	net->hiddenEnc  = AE_HIDDEN1;
	net->hiddenEnc2 = AE_HIDDEN2;
	net->mu         = latentDim;

	Matrix* enc1  = matrix_create(AE_HIDDEN1,   AE_INPUT_DIM);
	Matrix* enc1b = matrix_create(AE_HIDDEN1,   batchSize);
	Matrix* enc2  = matrix_create(AE_HIDDEN2,   AE_HIDDEN1);
	Matrix* enc2b = matrix_create(AE_HIDDEN2,   batchSize);
	Matrix* fc_mu  = matrix_create(latentDim,   AE_HIDDEN2);
	Matrix* fc_mub = matrix_create(latentDim,   batchSize);

	/* Kaiming uniform init for weights and biases — matches PyTorch nn.Linear */
	matrix_init(enc1,   AE_INPUT_DIM);
	matrix_init(enc1b,  AE_INPUT_DIM);
	matrix_init(enc2,   AE_HIDDEN1);
	matrix_init(enc2b,  AE_HIDDEN1);
	matrix_init(fc_mu,  AE_HIDDEN2);
	matrix_init(fc_mub, AE_HIDDEN2);

	net->hiddenWeightsEnc  = enc1;
	net->hiddenWeightsEnc2 = enc2;
	net->hiddenWeightsMu   = fc_mu;
	net->hiddenBiasEnc  = enc1b;
	net->hiddenBiasEnc2 = enc2b;
	net->hiddenBiasMu   = fc_mub;

	/* Decoder */
	net->hiddenDec  = AE_HIDDEN2;
	net->hiddenDec2 = AE_HIDDEN1;
	net->output     = AE_INPUT_DIM;

	Matrix* dec1   = matrix_create(AE_HIDDEN2,   latentDim);
	Matrix* dec2   = matrix_create(AE_HIDDEN1,   AE_HIDDEN2);
	Matrix* out_w  = matrix_create(AE_INPUT_DIM, AE_HIDDEN1);
	Matrix* dec1b  = matrix_create(AE_HIDDEN2,   batchSize);
	Matrix* dec2b  = matrix_create(AE_HIDDEN1,   batchSize);
	Matrix* out_wb = matrix_create(AE_INPUT_DIM, batchSize);

	matrix_init(dec1,   latentDim);
	matrix_init(dec1b,  latentDim);
	matrix_init(dec2,   AE_HIDDEN2);
	matrix_init(dec2b,  AE_HIDDEN2);
	matrix_init(out_w,  AE_HIDDEN1);
	matrix_init(out_wb, AE_HIDDEN1);

	net->hiddenWeightsDec  = dec1;
	net->hiddenWeightsDec2 = dec2;
	net->outputWeights     = out_w;
	net->hiddenBiasDec  = dec1b;
	net->hiddenBiasDec2 = dec2b;
	net->outputBias     = out_wb;
	return net;
}

static bool is_valid_double(double x) {
	return x * 0.0 == 0.0;
}

/*
 * adam_step — one Adam parameter update, fully in-place.
 *
 * Uses PyTorch-compatible bias correction:
 *   lr_t = lr * sqrt(1 - β2^t) / (1 - β1^t)
 *   param -= lr_t * m / (sqrt(v) + ε)
 *
 * Parallelised when n > 2048 (large weight matrices only).
 */
static void adam_step(double * restrict vd, double * restrict sd,
                      const double * restrict grad, double * restrict param,
                      size_t n, double lr, int t)
{
	double bc1  = 1.0 - pow(ADAM_BETA1, t);
	double bc2  = 1.0 - pow(ADAM_BETA2, t);
	double lr_t = lr * sqrt(bc2) / bc1;
#pragma omp parallel for if(n > 2048) schedule(static)
	for (size_t k = 0; k < n; k++) {
		double g = grad[k];
		vd[k] = ADAM_BETA1 * vd[k] + (1.0 - ADAM_BETA1) * g;
		sd[k] = ADAM_BETA2 * sd[k] + (1.0 - ADAM_BETA2) * g * g;
		param[k] -= lr_t * vd[k] / (sqrt(sd[k]) + ADAM_EPS);
	}
}

#define ADAM_UPDATE(vd, sd, grad, param)                                       \
	do {                                                                       \
		Matrix* _g = (grad);                                                   \
		adam_step((vd)->data, (sd)->data, _g->data, (param)->data,             \
		          (size_t)(vd)->rows * (vd)->cols, net->learning_rate, _step); \
		matrix_free(_g);                                                       \
	} while (0)


double network_train(NeuralNetwork* net, Matrix* X, int batch_size,
                     Matrix** vds, Matrix** sds, int _step) {
	/* ── Forward pass ─────────────────────────────────────────────────── */
	/* Encoder */
	Matrix* z0 = add(dot(net->hiddenWeightsEnc,  X,  0), net->hiddenBiasEnc,  1);
	Matrix* A0 = apply(relu, z0, 0);
	Matrix* z1 = add(dot(net->hiddenWeightsEnc2, A0, 0), net->hiddenBiasEnc2, 1);
	Matrix* A1 = apply(relu, z1, 0);
	Matrix* z2 = add(dot(net->hiddenWeightsMu,   A1, 0), net->hiddenBiasMu,   1);
	Matrix* A2 = apply(relu, z2, 0);
	/* Decoder */
	Matrix* z3 = add(dot(net->hiddenWeightsDec,  A2, 0), net->hiddenBiasDec,  1);
	Matrix* A3 = apply(relu, z3, 0);
	Matrix* z4 = add(dot(net->hiddenWeightsDec2, A3, 0), net->hiddenBiasDec2, 1);
	Matrix* A4 = apply(relu, z4, 0);
	Matrix* z5 = add(dot(net->outputWeights,     A4, 0), net->outputBias,     1);
	Matrix* A5 = apply(relu, z5, 0);

	/* ── MSE loss and output gradient ─────────────────────────────────── */
	/* Match PyTorch nn.MSELoss(reduction='mean'): loss = mean(diff²)
	 * Gradient: d_loss/d_output_i = 2*diff_i / n                        */
	double loss = 0.0;
	int n = X->rows * X->cols;   /* total elements = AE_INPUT_DIM * batch_size */
	Matrix* dA5 = matrix_create(AE_INPUT_DIM, batch_size);
	for (int i = 0; i < X->rows; i++) {
		for (int j = 0; j < X->cols; j++) {
			double diff = A5->entries[i][j] - X->entries[i][j];
			loss += diff * diff;
			dA5->entries[i][j] = 2.0 * diff / n;
		}
	}
	loss /= n;
	/* ── Backpropagation (output → encoder) ──────────────────────────── */
	/* Output layer */
	Matrix* primed = reluPrime(z5);
	Matrix* dzn    = multiply(dA5, primed, 0);
	matrix_free(primed);
	ADAM_UPDATE(vds[0], sds[0], dot_NT(dzn, A4, 0), net->outputWeights);
	ADAM_UPDATE(vds[6], sds[6], matrix_copy(dzn), net->outputBias);
	assert(is_valid_double(net->outputWeights->entries[0][0]));

	/* Decoder layer 2 */
	Matrix* prev = dot_TN(net->outputWeights, dzn, 0);
	primed = reluPrime(z4);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[1], sds[1], dot_NT(dzn, A3, 0), net->hiddenWeightsDec2);
	ADAM_UPDATE(vds[7], sds[7], matrix_copy(dzn), net->hiddenBiasDec2);

	/* Decoder layer 1 */
	prev   = dot_TN(net->hiddenWeightsDec2, dzn, 0);
	primed = reluPrime(z3);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[2], sds[2], dot_NT(dzn, A2, 0), net->hiddenWeightsDec);
	ADAM_UPDATE(vds[8], sds[8], matrix_copy(dzn), net->hiddenBiasDec);

	/* Latent (mu) layer */
	prev   = dot_TN(net->hiddenWeightsDec, dzn, 0);
	primed = reluPrime(z2);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[3], sds[3], dot_NT(dzn, A1, 0), net->hiddenWeightsMu);
	ADAM_UPDATE(vds[9], sds[9], matrix_copy(dzn), net->hiddenBiasMu);

	/* Encoder layer 2 */
	prev   = dot_TN(net->hiddenWeightsMu, dzn, 0);
	primed = reluPrime(z1);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[4], sds[4], dot_NT(dzn, A0, 0), net->hiddenWeightsEnc2);
	ADAM_UPDATE(vds[10], sds[10], matrix_copy(dzn), net->hiddenBiasEnc2);

	/* Encoder layer 1 */
	prev   = dot_TN(net->hiddenWeightsEnc2, dzn, 0);
	primed = reluPrime(z0);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[5], sds[5], dot_NT(dzn, X, 0), net->hiddenWeightsEnc);
	ADAM_UPDATE(vds[11], sds[11], matrix_copy(dzn), net->hiddenBiasEnc);

	/* ── Free intermediate activations ─────────────────────────────── */
	matrix_free(z0); matrix_free(A0);
	matrix_free(z1); matrix_free(A1);
	matrix_free(z2); matrix_free(A2);
	matrix_free(z3); matrix_free(A3);
	matrix_free(z4); matrix_free(A4);
	matrix_free(z5); matrix_free(A5);
	matrix_free(dA5);
	matrix_free(dzn);
	return loss;
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int training_size,
                              int batch_size, int epochs, int latent_dim) {
	int n_batches = training_size / batch_size;

	/* Pre-flatten all images into (AE_INPUT_DIM × batch_size) matrices */
	Matrix** batches = malloc(n_batches * sizeof(Matrix*));
	for (int j = 0; j < n_batches; j++) {
		batches[j] = matrix_create(AE_INPUT_DIM, batch_size);
		for (int i = j * batch_size; i < j * batch_size + batch_size; i++) {
			Matrix* flat = matrix_flatten(imgs[i]->img_data, 0);
			for (int k = 0; k < AE_INPUT_DIM; k++) {
				batches[j]->entries[k][0] = flat->entries[k][0];
			}
			matrix_free(flat);
		}
	}

	/* Adam optimiser state — 12 slots: weights[0-5] then biases[6-11].
	 * Indexed back-to-front (output layer = 0) to match backprop order:
	 *  [0] outputW   [1] dec2W  [2] dec1W  [3] muW   [4] enc2W  [5] enc1W
	 *  [6] outputB   [7] dec2B  [8] dec1B  [9] muB  [10] enc2B [11] enc1B
	 */
	Matrix** vds = malloc(12 * sizeof(Matrix*));
	Matrix** sds = malloc(12 * sizeof(Matrix*));

	vds[0]  = matrix_create(AE_INPUT_DIM, AE_HIDDEN1);
	vds[1]  = matrix_create(AE_HIDDEN1,   AE_HIDDEN2);
	vds[2]  = matrix_create(AE_HIDDEN2,   latent_dim);
	vds[3]  = matrix_create(latent_dim,   AE_HIDDEN2);
	vds[4]  = matrix_create(AE_HIDDEN2,   AE_HIDDEN1);
	vds[5]  = matrix_create(AE_HIDDEN1,   AE_INPUT_DIM);
	vds[6]  = matrix_create(AE_INPUT_DIM, batch_size);
	vds[7]  = matrix_create(AE_HIDDEN1,   batch_size);
	vds[8]  = matrix_create(AE_HIDDEN2,   batch_size);
	vds[9]  = matrix_create(latent_dim,   batch_size);
	vds[10] = matrix_create(AE_HIDDEN2,   batch_size);
	vds[11] = matrix_create(AE_HIDDEN1,   batch_size);

	for (int i = 0; i < 12; i++) {
		matrix_fill(vds[i], 0.0);
		sds[i] = matrix_copy(vds[i]);
	}

	FILE *lf = fopen("losses_c.csv", "w");
	if (lf) fprintf(lf, "batch,loss\n");

#define BAR_WIDTH 30
	struct timespec ep_start, ep_end;
	for (int k = 0; k < epochs; k++) {
		clock_gettime(CLOCK_MONOTONIC, &ep_start);
		double epoch_loss = 0.0;
		for (int i = 0; i < n_batches; i++) {
			double batch_loss = network_train(net, batches[i], batch_size, vds, sds,
			                                  k * n_batches + i + 1);
			epoch_loss += batch_loss;
			if (lf) fprintf(lf, "%d,%.8f\n", k * n_batches + i, batch_loss);

			/* ── progress bar ────────────────────────────────────────── */
			int done  = (i + 1) * BAR_WIDTH / n_batches;
			struct timespec now;
			clock_gettime(CLOCK_MONOTONIC, &now);
			double secs = (now.tv_sec  - ep_start.tv_sec)
			            + (now.tv_nsec - ep_start.tv_nsec) * 1e-9;
			fprintf(stderr,
			        "\rEpoch %d/%d  [%-*.*s]  %d/%d  loss: %.5f  %.1fs",
			        k + 1, epochs,
			        BAR_WIDTH, done, "##############################",
			        i + 1, n_batches,
			        epoch_loss / (i + 1),
			        secs);
		}
		clock_gettime(CLOCK_MONOTONIC, &ep_end);
		double ep_secs = (ep_end.tv_sec  - ep_start.tv_sec)
		               + (ep_end.tv_nsec - ep_start.tv_nsec) * 1e-9;
		/* Overwrite the progress bar with the final epoch summary */
		fprintf(stderr, "\r%-78s\r", "");
		printf("Epoch %d/%d — avg loss: %.6f — time: %.2f s\n",
		       k + 1, epochs, epoch_loss / n_batches, ep_secs);
	}
	if (lf) fclose(lf);

	/* Free batch data after all epochs complete */
	for (int i = 0; i < n_batches; i++) {
		matrix_free(batches[i]);
	}
	free(batches);

	for (int i = 0; i < 12; i++) {
		matrix_free(vds[i]);
		matrix_free(sds[i]);
	}
	free(vds);
	free(sds);
}

Img* network_predict(NeuralNetwork* net, Img* input_data) {
	Img* new_img = malloc(sizeof(Img));
	new_img->img_data = matrix_create(28, 28);
	Matrix* x = matrix_flatten(input_data->img_data, 0);  /* (784 × 1) */

	/* Forward pass */
	Matrix* z0 = add(dot(net->hiddenWeightsEnc,  x,  0), net->hiddenBiasEnc,  1);
	Matrix* A0 = apply(relu, z0, 0);
	Matrix* z1 = add(dot(net->hiddenWeightsEnc2, A0, 0), net->hiddenBiasEnc2, 1);
	Matrix* A1 = apply(relu, z1, 0);
	Matrix* z2 = add(dot(net->hiddenWeightsMu,   A1, 0), net->hiddenBiasMu,   1);
	Matrix* A2 = apply(relu, z2, 0);
	Matrix* z3 = add(dot(net->hiddenWeightsDec,  A2, 0), net->hiddenBiasDec,  1);
	Matrix* A3 = apply(relu, z3, 0);
	Matrix* z4 = add(dot(net->hiddenWeightsDec2, A3, 0), net->hiddenBiasDec2, 1);
	Matrix* A4 = apply(relu, z4, 0);
	Matrix* z5 = add(dot(net->outputWeights,     A4, 0), net->outputBias,     1);
	Matrix* A5 = apply(relu, z5, 0);

	/* Unflatten (784 × 1) → (28 × 28) */
	Matrix* result = matrix_unflatten(A5);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			new_img->img_data->entries[i][j] = result->entries[i][j];
		}
	}

	matrix_free(z0); matrix_free(A0);
	matrix_free(z1); matrix_free(A1);
	matrix_free(z2); matrix_free(A2);
	matrix_free(z3); matrix_free(A3);
	matrix_free(z4); matrix_free(A4);
	matrix_free(z5); matrix_free(A5);
	matrix_free(result);
	matrix_free(x);
	return new_img;
}

/* Build "dir/name" into buf safely */
static void make_path(char* buf, size_t bufsz,
                      const char* dir, const char* name) {
	snprintf(buf, bufsz, "%s/%s", dir, name);
}

void network_save(NeuralNetwork* net, const char* dir) {
	if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
		perror("network_save: mkdir");
		return;
	}
	char path[1024];

	make_path(path, sizeof(path), dir, "descriptor");
	FILE* f = fopen(path, "w");
	if (!f) { perror(path); return; }
	fprintf(f, "%d\n%d\n%d\n%d\n%d\n%d\n%d\n",
	        net->X, net->hiddenEnc, net->hiddenEnc2, net->mu,
	        net->hiddenDec, net->hiddenDec2, net->output);
	fclose(f);

#define SAVE_M(name, mat) \
	make_path(path, sizeof(path), dir, name); matrix_save(mat, path)

	SAVE_M("hiddenEnc",   net->hiddenWeightsEnc);
	SAVE_M("hiddenEnc2",  net->hiddenWeightsEnc2);
	SAVE_M("hiddenMu",    net->hiddenWeightsMu);
	SAVE_M("hiddenDec",   net->hiddenWeightsDec);
	SAVE_M("hiddenDec2",  net->hiddenWeightsDec2);
	SAVE_M("output",      net->outputWeights);
	SAVE_M("encBias",     net->hiddenBiasEnc);
	SAVE_M("enc2Bias",    net->hiddenBiasEnc2);
	SAVE_M("muBias",      net->hiddenBiasMu);
	SAVE_M("decBias",     net->hiddenBiasDec);
	SAVE_M("dec2Bias",    net->hiddenBiasDec2);
	SAVE_M("outputBias",  net->outputBias);
#undef SAVE_M

	printf("Network saved to '%s'\n", dir);
}

NeuralNetwork* network_load(const char* dir) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char path[1024];
	char entry[MAXCHAR];

	make_path(path, sizeof(path), dir, "descriptor");
	FILE* f = fopen(path, "r");
	if (!f) { perror(path); free(net); return NULL; }
	fgets(entry, MAXCHAR, f); net->X          = atoi(entry);
	fgets(entry, MAXCHAR, f); net->hiddenEnc  = atoi(entry);
	fgets(entry, MAXCHAR, f); net->hiddenEnc2 = atoi(entry);
	fgets(entry, MAXCHAR, f); net->mu         = atoi(entry);
	fgets(entry, MAXCHAR, f); net->hiddenDec  = atoi(entry);
	fgets(entry, MAXCHAR, f); net->hiddenDec2 = atoi(entry);
	fgets(entry, MAXCHAR, f); net->output     = atoi(entry);
	fclose(f);

#define LOAD_M(name, field) \
	make_path(path, sizeof(path), dir, name); net->field = matrix_load(path)

	LOAD_M("hiddenEnc",   hiddenWeightsEnc);
	LOAD_M("hiddenEnc2",  hiddenWeightsEnc2);
	LOAD_M("hiddenMu",    hiddenWeightsMu);
	LOAD_M("hiddenDec",   hiddenWeightsDec);
	LOAD_M("hiddenDec2",  hiddenWeightsDec2);
	LOAD_M("output",      outputWeights);
	LOAD_M("encBias",     hiddenBiasEnc);
	LOAD_M("enc2Bias",    hiddenBiasEnc2);
	LOAD_M("muBias",      hiddenBiasMu);
	LOAD_M("decBias",     hiddenBiasDec);
	LOAD_M("dec2Bias",    hiddenBiasDec2);
	LOAD_M("outputBias",  outputBias);
#undef LOAD_M

	printf("Network loaded from '%s'\n", dir);
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("Architecture : %d→%d→%d→%d→%d→%d→%d\n",
	       net->X, net->hiddenEnc, net->hiddenEnc2, net->mu,
	       net->hiddenDec, net->hiddenDec2, net->output);
	printf("Learning rate: %g\n", net->learning_rate);
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hiddenWeightsEnc);
	matrix_free(net->hiddenWeightsEnc2);
	matrix_free(net->hiddenWeightsMu);
	matrix_free(net->hiddenWeightsDec);
	matrix_free(net->hiddenWeightsDec2);
	matrix_free(net->outputWeights);
	matrix_free(net->hiddenBiasEnc);
	matrix_free(net->hiddenBiasEnc2);
	matrix_free(net->hiddenBiasMu);
	matrix_free(net->hiddenBiasDec);
	matrix_free(net->hiddenBiasDec2);
	matrix_free(net->outputBias);
	free(net);
	net = NULL;
}