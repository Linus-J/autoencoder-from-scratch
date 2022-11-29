#include "nn.h"
#include <sys/stat.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../util/ziggurat_inline.h"

#define MAXCHAR 1000


// 1) Maybe add seperate layer struct
// 2) Allow batch image input as X (784xm) (m=100)
// 3) Add another layer
// 4) Use better descent function
// 784, 512, 128, 512, 784
NeuralNetwork* aeCreate(int latentDim, double lr, int batchSize) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->learning_rate = lr;
	
	//Encoder
	net->X = 784;
	net->hiddenEnc = 512;
	net->mu = latentDim;
	Matrix* enc1 = matrix_create(512, 784);
	Matrix* enc1b = matrix_create(512, batchSize);
	Matrix* fc_mu = matrix_create(latentDim, 512);
	Matrix* fc_mub = matrix_create(latentDim, batchSize);

	matrix_fill(enc1b, 0.0);
	matrix_fill(fc_mub, 0.0);
	matrix_init(enc1, 784, 1);
	matrix_init(fc_mu, 512, 1);

	net->hiddenWeightsEnc = enc1;
	net->hiddenWeightsMu = fc_mu;
	net->hiddenBiasEnc = enc1b;
	net->hiddenBiasMu = fc_mub;

	//Decoder
	net->hiddenDec = 512;
	net->output = 784;

	Matrix* dec1 = matrix_create(512, latentDim);
	Matrix* output_layer = matrix_create(784, 512);
	Matrix* dec1b = matrix_create(512, batchSize);
	Matrix* output_layerb = matrix_create(784, batchSize);

	matrix_fill(dec1b, 0.0);
	matrix_fill(output_layerb, 0.0);
	matrix_init(dec1, latentDim, 1);
	matrix_init(output_layer, 512, 1);
	
	net->hiddenWeightsDec = dec1;
	net->outputWeights = output_layer;
	net->hiddenBiasDec = dec1b;
	net->outputBias = output_layerb;
	return net;
}

bool is_valid_double(double x)
{
    return x*0.0==0.0;
}

Matrix* reparameterise(Matrix* mu, Matrix* log_var){
	Matrix* std = matrix_create(log_var->rows, log_var->cols);
	int i,j;

	for (i = 0; i < log_var->rows; i++){	
		for (j = 0; j < log_var->cols; j++){
			std -> entries[i][j] = exp(0.5*(log_var -> entries[i][j]));
		}
	}

	Matrix* eps = matrix_create(log_var->rows, log_var->cols);

	for (i = 0; i < log_var->rows; i++){	
		for (j = 0; j < log_var->cols; j++){
			eps -> entries[i][j] = r4_nor_value();
		}
	}
	Matrix* multiplied = multiply(eps, std);
	Matrix* res = add(mu, multiplied);
	matrix_free(std);
	matrix_free(eps);
	matrix_free(multiplied);
	return res;
}

double network_train(NeuralNetwork* net, Matrix* X) {
	// Feed forward
	// Encode	
	Matrix* hiddenEncIn = dot(net->hiddenWeightsEnc, X);
	Matrix* z0 = add(hiddenEncIn,net->hiddenBiasEnc);
	Matrix* A0 = apply(relu, z0);

	Matrix* mu = dot(net->hiddenWeightsMu, A0);
	Matrix* z1 = add(mu,net->hiddenBiasMu);
	Matrix* A1 = apply(relu, z1);

	// Decode
	Matrix* hiddenDecIn = dot(net->hiddenWeightsDec, A1);
	Matrix* z2 = add(hiddenDecIn, net->hiddenBiasDec);
	Matrix* A2 = apply(relu, z2);

	Matrix* final_inputs = dot(net->outputWeights, A2);
	Matrix* z3 = add(final_inputs,net->outputBias);
	Matrix* A3 = apply(relu, z3);

	// Calc loss y^-y
	double loss = 0.0;

	Matrix* dA3 = matrix_create(784, 1);
	double temp = 0;
	for (int i = 0; i < X->rows; i++){
		for (int j = 0; j < X->cols; j++){
			loss += 0.5*pow(X->entries[i][j]-A3->entries[i][j],2);
			dA3 -> entries[i][j] = A3->entries[i][j]-X->entries[i][j];
		}
	}
	// printf("Feed forward done\n");
	// Backpropogation and parameter update

	//Check multiplication and dot, dot is standrad matmul and 


	//Output layer
	Matrix* primed_mat = reluPrime(z3);
	Matrix* dzn = multiply(dA3, primed_mat); //dz3 (also db3 whilst input is a vector)
	Matrix* transposed_mat = transpose(A2);
	Matrix* dot_mat = dot(dzn, transposed_mat); //dw3
	Matrix* scaled_mat = scale(-(net->learning_rate), dot_mat);
	Matrix* added_mat = add(net->outputWeights, scaled_mat);
	
	matrix_free(net->outputWeights);
	net->outputWeights = added_mat;
	
	matrix_free(scaled_mat);
	scaled_mat = scale(-(net->learning_rate), dzn);
	Matrix* added_matB = add(net->outputBias, scaled_mat);
	matrix_free(net->outputBias);
	net->outputBias = added_matB;

	matrix_free(primed_mat);
	
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);
	

	assert(is_valid_double(net-> outputWeights -> entries[0][0]));
	// printf("Output layer done\n");
	//Decoder layer
	transposed_mat = transpose(net->outputWeights);
	dot_mat = dot(transposed_mat, dzn);
	primed_mat = reluPrime(z2);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat); //dz2

	matrix_free(transposed_mat);
	matrix_free(dot_mat);

	transposed_mat = transpose(A1);
	dot_mat = dot(dzn, transposed_mat); //dw2
	scaled_mat = scale(-(net->learning_rate), dot_mat);
	added_mat = add(net->hiddenWeightsDec, scaled_mat);
	
	matrix_free(net->hiddenWeightsDec);
	net->hiddenWeightsDec = added_mat;
	
	matrix_free(scaled_mat);
	scaled_mat = scale(-(net->learning_rate), dzn);
	added_matB = add(net->hiddenBiasDec, scaled_mat);
	matrix_free(net->hiddenBiasDec);
	net->hiddenBiasDec = added_matB;

	matrix_free(primed_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);
	// printf("Decoder layer done\n");
	//Latent layer
	transposed_mat = transpose(net->hiddenWeightsDec);
	dot_mat = dot(transposed_mat, dzn);
	primed_mat = reluPrime(z1);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat); //dz1

	matrix_free(transposed_mat);
	matrix_free(dot_mat);

	transposed_mat = transpose(A0);
	dot_mat = dot(dzn, transposed_mat); //dw1
	scaled_mat = scale(-(net->learning_rate), dot_mat);
	added_mat = add(net->hiddenWeightsMu, scaled_mat);
	
	matrix_free(net->hiddenWeightsMu);
	net->hiddenWeightsMu = added_mat;
	
	matrix_free(scaled_mat);
	scaled_mat = scale(-(net->learning_rate), dzn);
	added_matB = add(net->hiddenBiasMu, scaled_mat);
	matrix_free(net->hiddenBiasMu);
	net->hiddenBiasMu = added_matB;

	matrix_free(primed_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);
	// printf("Latent layer done\n");
	//Encoding layer
	transposed_mat = transpose(net->hiddenWeightsMu);
	dot_mat = dot(transposed_mat, dzn);
	primed_mat = reluPrime(z0);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat); //dz0

	matrix_free(transposed_mat);
	matrix_free(dot_mat);

	transposed_mat = transpose(X);
	dot_mat = dot(dzn, transposed_mat); //dw0
	scaled_mat = scale(-(net->learning_rate), dot_mat);
	added_mat = add(net->hiddenWeightsEnc, scaled_mat);
	
	matrix_free(net->hiddenWeightsEnc);
	net->hiddenWeightsEnc = added_mat;
	
	matrix_free(scaled_mat);
	scaled_mat = scale(-(net->learning_rate), dzn);
	added_matB = add(net->hiddenBiasEnc, scaled_mat);
	matrix_free(net->hiddenBiasEnc);
	net->hiddenBiasEnc = added_matB;

	matrix_free(primed_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);
	// printf("Backprop done\n");
	// Free hidden matrices
	matrix_free(z0);
	matrix_free(z1);
	matrix_free(z2);
	matrix_free(z3);
	matrix_free(hiddenEncIn);
	matrix_free(A0);
	matrix_free(mu);
	matrix_free(A1);
	matrix_free(hiddenDecIn);
	matrix_free(A2);
	matrix_free(final_inputs);
	matrix_free(A3);
	matrix_free(dA3);
	matrix_free(dzn);
	return loss;
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
	r4_nor_setup();
	double loss = 0.0;
	for (int i = 0; i < batch_size; i++) {
		Matrix* img_data = matrix_flatten(imgs[i]->img_data, 0); // 0 = flatten to column vector
		loss = network_train(net, img_data);
		if (i % 100 == 0){
			printf("Img No. %d\n", i);
			printf("Loss. %f\n", loss);
		}
		matrix_free(img_data);
	}
}

Img* network_predict(NeuralNetwork* net, Img* input_data) {
	Img *new_img = malloc(sizeof(Img));
	new_img -> img_data =  matrix_create(28, 28);
	Matrix* img_data = matrix_flatten(input_data->img_data, 0);

	Matrix* hiddenEncIn = dot(net->hiddenWeightsEnc, img_data);
	Matrix* z0 = add(hiddenEncIn,net->hiddenBiasEnc);
	Matrix* A0 = apply(relu, z0);

	Matrix* mu = dot(net->hiddenWeightsMu, A0);
	Matrix* z1 = add(mu,net->hiddenBiasMu);
	Matrix* A1 = apply(relu, z1);

	// Decode
	Matrix* hiddenDecIn = dot(net->hiddenWeightsDec, A1);
	Matrix* z2 = add(hiddenDecIn, net->hiddenBiasDec);
	Matrix* A2 = apply(relu, z2);

	Matrix* final_inputs = dot(net->outputWeights, A2);
	Matrix* z3 = add(final_inputs,net->outputBias);
	Matrix* A3 = apply(relu, z3);

	matrix_print(final_inputs);
	//undo until here
	// //28x28 reconstructed image
	Matrix* result = matrix_unflatten(A3);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j <28; j++) {
			new_img->img_data->entries[i][j] = result->entries[i][j];
		}
	}
	matrix_free(z0);
	matrix_free(z1);
	matrix_free(z2);
	matrix_free(z3);
	matrix_free(hiddenEncIn);
	matrix_free(A0);
	matrix_free(mu);
	matrix_free(A1);
	matrix_free(hiddenDecIn);
	matrix_free(A2);
	matrix_free(final_inputs);
	matrix_free(A3);
	matrix_free(result);
	matrix_free(img_data);
	return new_img;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->X);
	fprintf(descriptor, "%d\n", net->hiddenEnc);
	fprintf(descriptor, "%d\n", net->mu);
	fprintf(descriptor, "%d\n", net->hiddenDec);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hiddenWeightsEnc, "hiddenEnc");
	matrix_save(net->hiddenWeightsMu, "hiddenMu");
	matrix_save(net->hiddenWeightsDec, "hiddenDec");
	matrix_save(net->outputWeights, "output");
	matrix_save(net->hiddenBiasEnc, "encBias");
	matrix_save(net->hiddenBiasMu, "muBias");
	matrix_save(net->hiddenBiasDec, "decBias");
	matrix_save(net->outputBias, "outputBias");
	printf("Successfully written to '%s'\n", file_string);
	chdir(".."); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->X = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hiddenEnc = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->mu = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hiddenDec = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hiddenWeightsEnc = matrix_load("hiddenEnc");
	net->hiddenWeightsMu = matrix_load("hiddenMu");
	net->hiddenWeightsDec = matrix_load("hiddenDec");
	net->outputWeights = matrix_load("output");

	net->hiddenBiasEnc = matrix_load("encBias");
	net->hiddenBiasMu = matrix_load("muBias");
	net->hiddenBiasDec = matrix_load("decBias");
	net->outputBias = matrix_load("outputBias");

	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hiddenWeightsEnc);
	matrix_free(net->hiddenWeightsMu);
	matrix_free(net->hiddenWeightsDec);
	matrix_free(net->outputWeights);
	matrix_free(net->hiddenBiasEnc);
	matrix_free(net->hiddenBiasMu);
	matrix_free(net->hiddenBiasDec);
	matrix_free(net->outputBias);
	free(net);
	net = NULL;
}