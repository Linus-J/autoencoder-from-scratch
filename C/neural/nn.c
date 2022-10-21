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

// 784, 512, 128, 512, 784
NeuralNetwork* aeCreate(int latentDim, double lr) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->learning_rate = lr;
	
	//Encoder
	net->input = 784;
	net->hiddenEnc = 512;
	net->mu = latentDim;
	Matrix* enc1 = matrix_create(512, 784);
	Matrix* fc_mu = matrix_create(latentDim, 512);

	matrix_init(enc1, 784, 1);
	matrix_init(fc_mu, 512, 1);

	net->hiddenWeightsEnc = enc1;
	net->hiddenWeightsMu = fc_mu;

	//Decoder
	net->hiddenDec = 512;
	net->output = 784;

	Matrix* dec1 = matrix_create(512, latentDim);
	Matrix* output_layer = matrix_create(784, 512);

	matrix_init(dec1, latentDim, 1);
	matrix_init(output_layer, 512, 1);
	
	net->hiddenWeightsDec = dec1;
	net->outputWeights = output_layer;
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

double network_train(NeuralNetwork* net, Matrix* input) {
	// Feed forward
	// Encode	
	Matrix* hiddenEncIn = dot(net->hiddenWeightsEnc, input);
	Matrix* hiddenEncOut = apply(relu, hiddenEncIn);

	Matrix* mu = dot(net->hiddenWeightsMu, hiddenEncOut);
	Matrix* z = apply(relu, mu);

	// Decode
	Matrix* hiddenDecIn = dot(net->hiddenWeightsDec, z);
	Matrix* hiddenDecOut = apply(relu, hiddenDecIn);

	Matrix* final_inputs = dot(net->outputWeights, hiddenDecOut);
	Matrix* final_outputs = apply(relu, final_inputs);

	double loss = 0.0;

	Matrix* output_errors = matrix_create(784, 1);
	double temp = 0;
	for (int i = 0; i < input->rows; i++){
		for (int j = 0; j < input->cols; j++){
			temp = input->entries[i][j]-final_outputs->entries[i][j];
			output_errors -> entries[i][j] = temp;
			loss += output_errors -> entries[i][j];
		}
	}

	Matrix* transposed_err_mat = transpose(net->outputWeights);
	Matrix* hiddenDec_errors = dot(transposed_err_mat, output_errors);
	matrix_free(transposed_err_mat);
	transposed_err_mat = transpose(net->hiddenWeightsDec);
	Matrix* hiddenMu_errors = dot(transposed_err_mat, hiddenDec_errors);
	matrix_free(transposed_err_mat);
	transposed_err_mat = transpose(net->hiddenWeightsMu);
	Matrix* hiddenEnc_errors = dot(transposed_err_mat, hiddenMu_errors);
	matrix_free(transposed_err_mat);

	Matrix* primed_mat = reluPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, primed_mat);
	Matrix* transposed_mat = transpose(hiddenDecOut);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->outputWeights, scaled_mat);
	matrix_free(net->outputWeights);
	net->outputWeights = added_mat;

	assert(is_valid_double(net-> outputWeights -> entries[0][0]));

	matrix_free(primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	primed_mat = reluPrime(hiddenDecOut);
	multiplied_mat = multiply(hiddenDec_errors, primed_mat);
	transposed_mat = transpose(z);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hiddenWeightsDec, scaled_mat);
	matrix_free(net->hiddenWeightsDec);
	net->hiddenWeightsDec = added_mat; 

	matrix_free(primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	primed_mat = reluPrime(z);
	multiplied_mat = multiply(hiddenMu_errors, primed_mat);
	transposed_mat = transpose(hiddenEncOut);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hiddenWeightsMu, scaled_mat);
	matrix_free(net->hiddenWeightsMu);
	net->hiddenWeightsMu = added_mat; 

	matrix_free(primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	primed_mat = reluPrime(hiddenEncOut);
	multiplied_mat = multiply(hiddenEnc_errors, primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hiddenWeightsEnc, scaled_mat);
	matrix_free(net->hiddenWeightsEnc);
	net->hiddenWeightsEnc = added_mat;

	matrix_free(primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// Free hidden matrices
	matrix_free(hiddenEncIn);
	matrix_free(hiddenEncOut);
	matrix_free(mu);
	matrix_free(z);
	matrix_free(hiddenDecIn);
	matrix_free(hiddenDecOut);
	matrix_free(final_inputs);
	matrix_free(final_outputs);

	// Free error matrices
	matrix_free(hiddenEnc_errors);
	matrix_free(hiddenMu_errors);
	matrix_free(hiddenDec_errors);
	matrix_free(output_errors);
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
	Matrix* hiddenEncOut = apply(relu, hiddenEncIn);

	Matrix* mu = dot(net->hiddenWeightsMu, hiddenEncOut);
	Matrix* z = apply(relu, mu);

	Matrix* hiddenDecIn = dot(net->hiddenWeightsDec, z);
	Matrix* hiddenDecOut = apply(relu, hiddenDecIn);

	Matrix* final_inputs = dot(net->outputWeights, hiddenDecOut);
	Matrix* final_outputs = apply(relu, final_inputs);

	// //28x28 reconstructed image
	Matrix* result = matrix_unflatten(final_outputs);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j <28; j++) {
			new_img->img_data->entries[i][j] = result->entries[i][j];
		}
	}
	matrix_free(hiddenEncIn);
	matrix_free(hiddenEncOut);
	matrix_free(mu);
	matrix_free(z);
	matrix_free(hiddenDecIn);
	matrix_free(hiddenDecOut);
	matrix_free(final_inputs);
	matrix_free(final_outputs);
	matrix_free(result);
	matrix_free(img_data);
	return new_img;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hiddenEnc);
	fprintf(descriptor, "%d\n", net->mu);
	fprintf(descriptor, "%d\n", net->hiddenDec);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hiddenWeightsEnc, "hiddenEnc");
	matrix_save(net->hiddenWeightsMu, "hiddenMu");
	matrix_save(net->hiddenWeightsDec, "hiddenDec");
	matrix_save(net->outputWeights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir(".."); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
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
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hiddenWeightsEnc);
	matrix_free(net->hiddenWeightsMu);
	matrix_free(net->hiddenWeightsDec);
	matrix_free(net->outputWeights);
	free(net);
	net = NULL;
}