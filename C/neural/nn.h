#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"

typedef struct {
	int X;
	int hiddenEnc;
	int hiddenEnc2;
	int mu;
	int hiddenDec;
	int hiddenDec2;
	int output;
	double learning_rate;
	Matrix* hiddenWeightsEnc;
	Matrix* hiddenBiasEnc;
	Matrix* hiddenWeightsEnc2;
	Matrix* hiddenBiasEnc2;
	Matrix* hiddenWeightsMu;
	Matrix* hiddenBiasMu;
	Matrix* hiddenWeightsDec;
	Matrix* hiddenBiasDec;
	Matrix* hiddenWeightsDec2;
	Matrix* hiddenBiasDec2;
	Matrix* outputWeights;
	Matrix* outputBias;
} NeuralNetwork;

NeuralNetwork* aeCreate(int latentDim, double lr, int batchSize);
double         network_train(NeuralNetwork* net, Matrix* input_data,
                             int batch_size, Matrix** vds, Matrix** sds);
void           network_train_batch_imgs(NeuralNetwork* net, Img** imgs,
                             int training_size, int batch_size,
                             int epochs, int latent_dim);
Img*           network_predict(NeuralNetwork* net, Img* input_data);
void           network_save(NeuralNetwork* net, const char* dir);
NeuralNetwork* network_load(const char* dir);
void           network_print(NeuralNetwork* net);
void           network_free(NeuralNetwork* net);
