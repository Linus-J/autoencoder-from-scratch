#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"

typedef struct {
	int input;
	int hiddenEnc;
	int mu;
	int hiddenDec;
	int output;
	double learning_rate;
	Matrix* hiddenWeightsEnc;
	Matrix* hiddenWeightsMu;
	Matrix* hiddenWeightsDec;
	Matrix* outputWeights;
} NeuralNetwork;

// NeuralNetwork* network_create(int input, int hidden, int output, double lr);
NeuralNetwork* aeCreate(int latentDim, double lr);
Matrix* reparameterise(Matrix* mu, Matrix* log_var);
double network_train(NeuralNetwork* net, Matrix* input_data);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
//Matrix* network_predict_img(NeuralNetwork* net, Img* img);
//double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Img* network_predict(NeuralNetwork* net, Img* input_data);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);