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

	r4_nor_setup();  /* Initialise ziggurat RNG once before weight init */

	Matrix* enc1  = matrix_create(AE_HIDDEN1,   AE_INPUT_DIM);
	Matrix* enc1b = matrix_create(AE_HIDDEN1,   batchSize);
	Matrix* enc2  = matrix_create(AE_HIDDEN2,   AE_HIDDEN1);
	Matrix* enc2b = matrix_create(AE_HIDDEN2,   batchSize);
	Matrix* fc_mu  = matrix_create(latentDim,   AE_HIDDEN2);
	Matrix* fc_mub = matrix_create(latentDim,   batchSize);

	matrix_fill(enc1b,  0.0);
	matrix_fill(enc2b,  0.0);
	matrix_fill(fc_mub, 0.0);
	matrix_init(enc1,  AE_INPUT_DIM, 1);
	matrix_init(enc2,  AE_HIDDEN1,   1);
	matrix_init(fc_mu, AE_HIDDEN2,   1);

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

	matrix_fill(dec1b,  0.0);
	matrix_fill(dec2b,  0.0);
	matrix_fill(out_wb, 0.0);
	matrix_init(dec1,  latentDim,  1);
	matrix_init(dec2,  AE_HIDDEN2, 1);
	matrix_init(out_w, AE_HIDDEN1, 1);

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
 * ADAM_UPDATE — one Adam parameter update in place.
 *
 * vd/sd   first/second moment accumulators (replaced in place)
 * grad    gradient matrix (may be consumed by sub-expressions)
 * param   network weight/bias (replaced in place)
 *
 * The macro wraps each expression in its own statement so the caller
 * just writes:  ADAM_UPDATE(vds[i], sds[i], grad_expr, net->someWeight);
 */
#define ADAM_UPDATE(vd, sd, grad, param)                                       \
	do {                                                                       \
		Matrix* _g      = (grad);                                              \
		Matrix* _vd_new = add(scale(ADAM_BETA1,           (vd),     0),       \
		                      scale(1.0 - ADAM_BETA1,     _g,       0), 3);   \
		matrix_free(vd); (vd) = _vd_new;                                       \
		Matrix* _sd_new = add(scale(ADAM_BETA2,           (sd),     0),       \
		                      scale(1.0 - ADAM_BETA2, sqrm(_g, 1),  1), 3);   \
		matrix_free(sd); (sd) = _sd_new;                                       \
		Matrix* _step = scale(-(net->learning_rate),                           \
		                      divide((vd),                                     \
		                             sqrtm(addScalar(ADAM_EPS, (sd), 0), 1),   \
		                             2), 1);                                   \
		Matrix* _new = add((param), _step, 2);                                 \
		matrix_free(param); (param) = _new;                                    \
	} while (0)


double network_train(NeuralNetwork* net, Matrix* X, int batch_size,
                     Matrix** vds, Matrix** sds) {
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
	double loss = 0.0;
	Matrix* dA5 = matrix_create(AE_INPUT_DIM, batch_size);
	for (int i = 0; i < X->rows; i++) {
		for (int j = 0; j < X->cols; j++) {
			double diff = A5->entries[i][j] - X->entries[i][j];
			loss += 0.5 * diff * diff;
			dA5->entries[i][j] = diff;
		}
	}

	/* ── Backpropagation (output → encoder) ──────────────────────────── */
	/* Output layer */
	Matrix* primed = reluPrime(z5);
	Matrix* dzn    = multiply(dA5, primed, 0);
	matrix_free(primed);
	ADAM_UPDATE(vds[0], sds[0], dot(dzn, transpose(A4, 0), 2), net->outputWeights);
	ADAM_UPDATE(vds[6], sds[6], matrix_copy(dzn), net->outputBias);
	assert(is_valid_double(net->outputWeights->entries[0][0]));

	/* Decoder layer 2 */
	Matrix* prev = dot(transpose(net->outputWeights, 0), dzn, 1);
	primed = reluPrime(z4);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[1], sds[1], dot(dzn, transpose(A3, 0), 2), net->hiddenWeightsDec2);
	ADAM_UPDATE(vds[7], sds[7], matrix_copy(dzn), net->hiddenBiasDec2);

	/* Decoder layer 1 */
	prev   = dot(transpose(net->hiddenWeightsDec2, 0), dzn, 1);
	primed = reluPrime(z3);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[2], sds[2], dot(dzn, transpose(A2, 0), 2), net->hiddenWeightsDec);
	ADAM_UPDATE(vds[8], sds[8], matrix_copy(dzn), net->hiddenBiasDec);

	/* Latent (mu) layer */
	prev   = dot(transpose(net->hiddenWeightsDec, 0), dzn, 1);
	primed = reluPrime(z2);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[3], sds[3], dot(dzn, transpose(A1, 0), 2), net->hiddenWeightsMu);
	ADAM_UPDATE(vds[9], sds[9], matrix_copy(dzn), net->hiddenBiasMu);

	/* Encoder layer 2 */
	prev   = dot(transpose(net->hiddenWeightsMu, 0), dzn, 1);
	primed = reluPrime(z1);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[4], sds[4], dot(dzn, transpose(A0, 0), 2), net->hiddenWeightsEnc2);
	ADAM_UPDATE(vds[10], sds[10], matrix_copy(dzn), net->hiddenBiasEnc2);

	/* Encoder layer 1 */
	prev   = dot(transpose(net->hiddenWeightsEnc2, 0), dzn, 1);
	primed = reluPrime(z0);
	matrix_free(dzn);
	dzn = multiply(prev, primed, 0);
	matrix_free(prev); matrix_free(primed);
	ADAM_UPDATE(vds[5], sds[5], dot(dzn, transpose(X, 0), 2), net->hiddenWeightsEnc);
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

	struct timespec ep_start, ep_end;
	for (int k = 0; k < epochs; k++) {
		clock_gettime(CLOCK_MONOTONIC, &ep_start);
		double epoch_loss = 0.0;
		for (int i = 0; i < n_batches; i++) {
			epoch_loss += network_train(net, batches[i], batch_size, vds, sds);
		}
		clock_gettime(CLOCK_MONOTONIC, &ep_end);
		double ep_secs = (ep_end.tv_sec  - ep_start.tv_sec)
		               + (ep_end.tv_nsec - ep_start.tv_nsec) * 1e-9;
		printf("Epoch %d/%d — avg loss: %.6f — time: %.2f s\n",
		       k + 1, epochs, epoch_loss / n_batches, ep_secs);
	}

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