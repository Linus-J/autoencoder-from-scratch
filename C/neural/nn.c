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

// 784, 512, 256, 128, 256, 512, 784
NeuralNetwork* aeCreate(int latentDim, double lr, int batchSize) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->learning_rate = lr;
	
	//Encoder
	net->X = 784;
	net->hiddenEnc = 512;
	net->hiddenEnc2 = 256;
	net->mu = latentDim;
	Matrix* enc1 = matrix_create(512, 784);
	Matrix* enc1b = matrix_create(512, batchSize);
	Matrix* enc2 = matrix_create(256, 512);
	Matrix* enc2b = matrix_create(256, batchSize);
	Matrix* fc_mu = matrix_create(latentDim, 256);
	Matrix* fc_mub = matrix_create(latentDim, batchSize);

	matrix_fill(enc1b, 0.0);
	matrix_fill(enc2b, 0.0);
	matrix_fill(fc_mub, 0.0);
	matrix_init(enc1, 784, 1);
	matrix_init(enc2, 512, 1);
	matrix_init(fc_mu, 256, 1);

	net->hiddenWeightsEnc = enc1;
	net->hiddenWeightsEnc2 = enc2;
	net->hiddenWeightsMu = fc_mu;
	net->hiddenBiasEnc = enc1b;
	net->hiddenBiasEnc2 = enc2b;
	net->hiddenBiasMu = fc_mub;

	//Decoder
	net->hiddenDec = 256;
	net->hiddenDec2 = 512;
	net->output = 784;

	Matrix* dec1 = matrix_create(256, latentDim);
	Matrix* dec2 = matrix_create(512, 256);
	Matrix* output_layer = matrix_create(784, 512);
	Matrix* dec1b = matrix_create(256, batchSize);
	Matrix* dec2b = matrix_create(512, batchSize);
	Matrix* output_layerb = matrix_create(784, batchSize);

	matrix_fill(dec1b, 0.0);
	matrix_fill(dec2b, 0.0);
	matrix_fill(output_layerb, 0.0);
	matrix_init(dec1, latentDim, 1);
	matrix_init(dec2, 256, 1);
	matrix_init(output_layer, 512, 1);
	
	net->hiddenWeightsDec = dec1;
	net->hiddenWeightsDec2 = dec2;
	net->outputWeights = output_layer;
	net->hiddenBiasDec = dec1b;
	net->hiddenBiasDec2 = dec2b;
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
	Matrix* multiplied = multiply(eps, std, 0);
	Matrix* res = add(mu, multiplied, 0);
	matrix_free(std);
	matrix_free(eps);
	matrix_free(multiplied);
	return res;
}

double network_train(NeuralNetwork* net, Matrix* X, int batch_size, Matrix** vds, Matrix** sds) {
	// Feed forward
	// Encode	
	Matrix* z0 = add(dot(net->hiddenWeightsEnc, X, 0),net->hiddenBiasEnc, 1);
	Matrix* A0 = apply(relu, z0, 0);

	Matrix* z1 = add(dot(net->hiddenWeightsEnc2, A0, 0),net->hiddenBiasEnc2, 1);
	Matrix* A1 = apply(relu, z1, 0);

	Matrix* z2 = add(dot(net->hiddenWeightsMu, A1, 0),net->hiddenBiasMu, 1);
	Matrix* A2 = apply(relu, z2, 0);

	// Decode
	Matrix* z3 = add(dot(net->hiddenWeightsDec, A2, 0), net->hiddenBiasDec, 1);
	Matrix* A3 = apply(relu, z3, 0);

	Matrix* z4 = add(dot(net->hiddenWeightsDec2, A3, 0), net->hiddenBiasDec2, 1);
	Matrix* A4 = apply(relu, z4, 0);

	Matrix* z5 = add(dot(net->outputWeights, A4, 0),net->outputBias, 1);
	Matrix* A5 = apply(relu, z5, 0);

	// Calc loss y^-y
	double loss = 0.0;
	Matrix* dA5 = matrix_create(784, batch_size);
	for (int i = 0; i < X->rows; i++){
		for (int j = 0; j < X->cols; j++){
			loss += 0.5*pow(X->entries[i][j]-A5->entries[i][j],2);
			dA5 -> entries[i][j] = A5->entries[i][j]-X->entries[i][j];
		}
	}

	/* Backpropogation and parameter update */

	//Output layer
	Matrix* primed_mat = reluPrime(z5);
	Matrix* dzn = multiply(dA5, primed_mat, 0); //dz5 (also db5 whilst input is a vector)
	Matrix* dot_mat = dot(dzn, transpose(A4, 0), 2); //dw5
	//Update W5
	Matrix* tempM = add(scale(0.9,vds[0], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[0]);
	vds[0] = tempM;
	tempM = add(scale(0.999,sds[0], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[0]);
	sds[0] = tempM;	
	Matrix* added_mat = add(net->outputWeights, scale(-(net->learning_rate), divide(vds[0],sqrtm(addScalar(0.00000001,sds[0],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->outputWeights);
	net->outputWeights = added_mat;
	//Update b5
	tempM = add(scale(0.9,vds[6], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[6]);
	vds[6] = tempM;
	tempM = add(scale(0.999,sds[6], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[6]);
	sds[6] = tempM;
	Matrix* added_matB = add(net->outputBias, scale(-(net->learning_rate), divide(vds[6],sqrtm(addScalar(0.00000001,sds[6],0),1),2), 1), 2); //Update bias with ADAM optimiser

	matrix_free(net->outputBias);
	net->outputBias = added_matB;
	matrix_free(primed_mat);
	matrix_free(dot_mat);	
	assert(is_valid_double(net-> outputWeights -> entries[0][0]));

	//Decoder2 layer
	dot_mat = dot(transpose(net->outputWeights, 0), dzn, 1);
	primed_mat = reluPrime(z4);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat, 0); //dz4
	matrix_free(dot_mat);
	dot_mat = dot(dzn, transpose(A3, 0), 2); //dw4
	//Update W4
	tempM = add(scale(0.9,vds[1], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[1]);
	vds[1] = tempM;
	tempM = add(scale(0.999,sds[1], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[1]);
	sds[1] = tempM;
	added_mat = add(net->hiddenWeightsDec2, scale(-(net->learning_rate), divide(vds[1],sqrtm(addScalar(0.00000001,sds[1],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->hiddenWeightsDec2);
	net->hiddenWeightsDec2 = added_mat;
	//Update b4
	tempM = add(scale(0.9,vds[7], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[7]);
	vds[7] = tempM;
	tempM = add(scale(0.999,sds[7], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[7]);
	sds[7] = tempM;
	added_matB = add(net->hiddenBiasDec2, scale(-(net->learning_rate), divide(vds[7],sqrtm(addScalar(0.00000001,sds[7],0),1),2), 1), 2); //Update bias with ADAM optimiser
	matrix_free(net->hiddenBiasDec2);
	net->hiddenBiasDec2 = added_matB;
	matrix_free(primed_mat);
	matrix_free(dot_mat);

	//Decoder layer
	dot_mat = dot(transpose(net->hiddenWeightsDec2, 0), dzn, 1);
	primed_mat = reluPrime(z3);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat, 0); //dz3
	matrix_free(dot_mat);
	dot_mat = dot(dzn, transpose(A2, 0), 2); //dw3
	//Update W3
	tempM = add(scale(0.9,vds[2], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[2]);
	vds[2] = tempM;
	tempM = add(scale(0.999,sds[2], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[2]);
	sds[2] = tempM;
	added_mat = add(net->hiddenWeightsDec, scale(-(net->learning_rate), divide(vds[2],sqrtm(addScalar(0.00000001,sds[2],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->hiddenWeightsDec);
	net->hiddenWeightsDec = added_mat;
	//Update b3
	tempM = add(scale(0.9,vds[8], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[8]);
	vds[8] = tempM;
	tempM = add(scale(0.999,sds[8], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[8]);
	sds[8] = tempM;
	added_matB = add(net->hiddenBiasDec, scale(-(net->learning_rate), divide(vds[8],sqrtm(addScalar(0.00000001,sds[8],0),1),2), 1), 2); //Update bias with ADAM optimiser
	matrix_free(net->hiddenBiasDec);
	net->hiddenBiasDec = added_matB;
	matrix_free(primed_mat);
	matrix_free(dot_mat);

	//Latent layer
	dot_mat = dot(transpose(net->hiddenWeightsDec, 0), dzn, 1);
	primed_mat = reluPrime(z2);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat, 0); //dz2
	matrix_free(dot_mat);
	dot_mat = dot(dzn, transpose(A1, 0), 2); //dw2
	//Update W2
	tempM = add(scale(0.9,vds[3], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[3]);
	vds[3] = tempM;
	tempM = add(scale(0.999,sds[3], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[3]);
	sds[3] = tempM;
	added_mat = add(net->hiddenWeightsMu, scale(-(net->learning_rate), divide(vds[3],sqrtm(addScalar(0.00000001,sds[3],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->hiddenWeightsMu);
	net->hiddenWeightsMu = added_mat;
	//Update b2
	tempM = add(scale(0.9,vds[9], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[9]);
	vds[9] = tempM;
	tempM = add(scale(0.999,sds[9], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[9]);
	sds[9] = tempM;
	added_matB = add(net->hiddenBiasMu, scale(-(net->learning_rate), divide(vds[9],sqrtm(addScalar(0.00000001,sds[9],0),1),2), 1), 2); //Update bias with ADAM optimiser
	matrix_free(net->hiddenBiasMu);
	net->hiddenBiasMu = added_matB;
	matrix_free(primed_mat);
	matrix_free(dot_mat);

	//Encoding2 layer
	dot_mat = dot(transpose(net->hiddenWeightsMu, 0), dzn, 1);
	primed_mat = reluPrime(z1);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat, 0); //dz1
	matrix_free(dot_mat);
	dot_mat = dot(dzn, transpose(A0, 0), 2); //dw1
	//Update W1
	tempM = add(scale(0.9,vds[4], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[4]);
	vds[4] = tempM;
	tempM = add(scale(0.999,sds[4], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[4]);
	sds[4] = tempM;
	added_mat = add(net->hiddenWeightsEnc2, scale(-(net->learning_rate), divide(vds[4],sqrtm(addScalar(0.00000001,sds[4],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->hiddenWeightsEnc2);
	net->hiddenWeightsEnc2 = added_mat;
	//Update b1
	tempM = add(scale(0.9,vds[10], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[10]);
	vds[10] = tempM;
	tempM = add(scale(0.999,sds[10], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[10]);
	sds[10] = tempM;
	added_matB = add(net->hiddenBiasEnc2, scale(-(net->learning_rate), divide(vds[10],sqrtm(addScalar(0.00000001,sds[10],0),1),2), 1), 2); //Update bias with ADAM optimiser
	matrix_free(net->hiddenBiasEnc2);
	net->hiddenBiasEnc2 = added_matB;
	matrix_free(primed_mat);
	matrix_free(dot_mat);

	//Encoding layer
	dot_mat = dot(transpose(net->hiddenWeightsEnc2, 0), dzn, 1);
	primed_mat = reluPrime(z0);
	matrix_free(dzn);
	dzn = multiply(dot_mat, primed_mat, 0); //dz0
	matrix_free(dot_mat);
	dot_mat = dot(dzn, transpose(X, 0), 2); //dw0
	//Update W0
	tempM = add(scale(0.9,vds[5], 0),scale(0.1,dot_mat, 0), 3);
	matrix_free(vds[5]);
	vds[5] = tempM;
	tempM = add(scale(0.999,sds[5], 0),scale(0.001, sqrm(dot_mat, 0), 1), 3);
	matrix_free(sds[5]);
	sds[5] = tempM;
	added_mat = add(net->hiddenWeightsEnc, scale(-(net->learning_rate), divide(vds[5],sqrtm(addScalar(0.00000001,sds[5],0),1),2), 1), 2); //Update weights with ADAM optimiser
	matrix_free(net->hiddenWeightsEnc);
	net->hiddenWeightsEnc = added_mat;
	//Update b1
	tempM = add(scale(0.9,vds[11], 0),scale(0.1,dzn, 0), 3);
	matrix_free(vds[11]);
	vds[11] = tempM;
	tempM = add(scale(0.999,sds[11], 0),scale(0.001, sqrm(dzn, 0), 1), 3);
	matrix_free(sds[11]);
	sds[11] = tempM;
	added_matB = add(net->hiddenBiasEnc, scale(-(net->learning_rate), divide(vds[11],sqrtm(addScalar(0.00000001,sds[11],0),1),2), 1), 2); //Update bias with ADAM optimiser
	matrix_free(net->hiddenBiasEnc);
	net->hiddenBiasEnc = added_matB;

	matrix_free(primed_mat);
	matrix_free(dot_mat);

	// Free hidden matrices
	matrix_free(z0);
	matrix_free(z1);
	matrix_free(z2);
	matrix_free(z3);
	matrix_free(z4);
	matrix_free(z5);
	matrix_free(A0);
	matrix_free(A1);
	matrix_free(A2);
	matrix_free(A3);
	matrix_free(A4);
	matrix_free(A5);
	matrix_free(dA5);
	matrix_free(dzn);
	return loss;
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int training_size, int batch_size, int epochs, int latent_dim) {
	r4_nor_setup();
	double loss = 0.0;
	Matrix** batches = malloc(training_size/batch_size * sizeof(Matrix*));
	for (int j=0; j<training_size/batch_size; j++){
		batches[j] = matrix_create(784, batch_size);
		
		for (int i = j*batch_size; i < j*batch_size+batch_size; i++) {
			Matrix* img_data = matrix_flatten(imgs[i]->img_data, 0);
			for (int k=0; k<784; k++) {
				batches[j] -> entries[k][0] = img_data-> entries[k][0];
			}
			matrix_free(img_data);
		}
	}
	// 784, 512, 256, 128, 256, 512, 784
	//vds handle momentum for Adam optimiser
	Matrix** vds = malloc(12 * sizeof(Matrix*));
	vds[0] = matrix_create(784, 512);
	vds[1] = matrix_create(512, 256);
	vds[2] = matrix_create(256, latent_dim);
	vds[3] = matrix_create(latent_dim, 256);
	vds[4] = matrix_create(256, 512);
	vds[5] = matrix_create(512, 784);
	matrix_fill(vds[0], 0.0);
	matrix_fill(vds[1], 0.0);
	matrix_fill(vds[2], 0.0);
	matrix_fill(vds[3], 0.0);
	matrix_fill(vds[4], 0.0);
	matrix_fill(vds[5], 0.0);
	vds[11]= matrix_create(512, batch_size);
	vds[10] = matrix_create(256, batch_size);
	vds[9] = matrix_create(latent_dim, batch_size);
	vds[8]= matrix_create(256, batch_size);
	vds[7] = matrix_create(512, batch_size);
	vds[6] = matrix_create(784, batch_size); 
	matrix_fill(vds[6], 0.0);
	matrix_fill(vds[7], 0.0);
	matrix_fill(vds[8], 0.0);
	matrix_fill(vds[9], 0.0);
	matrix_fill(vds[10], 0.0);
	matrix_fill(vds[11], 0.0);
	//vds handle RMSprop for Adam optimiser
	Matrix** sds = malloc(12 * sizeof(Matrix*));
	sds[0] = matrix_create(784, 512);
	sds[1] = matrix_create(512, 256);
	sds[2] = matrix_create(256, latent_dim);
	sds[3] = matrix_create(latent_dim, 256);
	sds[4] = matrix_create(256, 512);
	sds[5] = matrix_create(512, 784);
	matrix_fill(sds[0], 0.0);
	matrix_fill(sds[1], 0.0);
	matrix_fill(sds[2], 0.0);
	matrix_fill(sds[3], 0.0);
	matrix_fill(sds[4], 0.0);
	matrix_fill(sds[5], 0.0);
	sds[11]= matrix_create(512, batch_size);
	sds[10] = matrix_create(256, batch_size);
	sds[9] = matrix_create(latent_dim, batch_size);
	sds[8]= matrix_create(256, batch_size);
	sds[7] = matrix_create(512, batch_size);
	sds[6] = matrix_create(784, batch_size); 
	matrix_fill(sds[6], 0.0);
	matrix_fill(sds[7], 0.0);
	matrix_fill(sds[8], 0.0);
	matrix_fill(sds[9], 0.0);
	matrix_fill(sds[10], 0.0);
	matrix_fill(sds[11], 0.0);

	for (int k = 0; k < epochs; k++) {
		printf("Epoch. %d\n", k);
		for (int i = 0; i < training_size/batch_size; i++) {
			loss = network_train(net, batches[i], batch_size, vds, sds);
			printf("Batch No. %d\n", i);
			printf("Loss. %f\n", loss);
			matrix_free(batches[i]);
		}
	}
	free(batches);
	batches = NULL;
	for (int i=0; i<12; i++){
		matrix_free(sds[i]);
		matrix_free(vds[i]);
	}
	free(sds);
	free(vds);
	sds = NULL;
	vds = NULL;
}

Img* network_predict(NeuralNetwork* net, Img* input_data, int batch_size) {
	Img *new_img = malloc(sizeof(Img));
	new_img -> img_data =  matrix_create(28, 28);
	Matrix* img_mat = matrix_flatten(input_data->img_data, 0);
	Matrix* mymat = matrix_create(784,batch_size);
	for (int i = 0; i < batch_size; i++) {
		for (int j=0; j<784; j++) {
			if (i==0){
				mymat-> entries[j][0] = img_mat-> entries[j][0];
			}
			else{
				mymat-> entries[j][i] = 0;
			}
		}
	}

	Matrix* z0 = add(dot(net->hiddenWeightsEnc, mymat, 0),net->hiddenBiasEnc, 1);
	Matrix* A0 = apply(relu, z0, 0);

	Matrix* z1 = add(dot(net->hiddenWeightsEnc2, A0, 0),net->hiddenBiasEnc2, 1);
	Matrix* A1 = apply(relu, z1, 0);

	Matrix* z2 = add(dot(net->hiddenWeightsMu, A1, 0),net->hiddenBiasMu, 1);
	Matrix* A2 = apply(relu, z2, 0);

	// Decode
	Matrix* z3 = add(dot(net->hiddenWeightsDec, A2, 0), net->hiddenBiasDec, 1);
	Matrix* A3 = apply(relu, z3, 0);

	Matrix* z4 = add(dot(net->hiddenWeightsDec2, A3, 0), net->hiddenBiasDec2, 1);
	Matrix* A4 = apply(relu, z4, 0);

	Matrix* z5 = add(dot(net->outputWeights, A4, 0),net->outputBias, 1);
	Matrix* A5 = apply(relu, z5, 0);

	//28x28 reconstructed image
	Matrix* recon = matrix_create(784,1);

	for (int j=0; j<784; j++) {
		recon -> entries[j][0] = A5-> entries[j][0];
	}

	Matrix* result = matrix_unflatten(recon);
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j <28; j++) {
			new_img->img_data->entries[i][j] = result->entries[i][j];
		}
	}
	matrix_free(z0);
	matrix_free(z1);
	matrix_free(z2);
	matrix_free(z3);
	matrix_free(z4);
	matrix_free(z5);
	matrix_free(A0);
	matrix_free(A1);
	matrix_free(A2);
	matrix_free(A3);
	matrix_free(A4);
	matrix_free(A5);
	matrix_free(result);
	matrix_free(mymat);
	matrix_free(img_mat);
	matrix_free(recon);
	return new_img;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->X);
	fprintf(descriptor, "%d\n", net->hiddenEnc);
	fprintf(descriptor, "%d\n", net->hiddenEnc2);
	fprintf(descriptor, "%d\n", net->mu);
	fprintf(descriptor, "%d\n", net->hiddenDec);
	fprintf(descriptor, "%d\n", net->hiddenDec2);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hiddenWeightsEnc, "hiddenEnc");
	matrix_save(net->hiddenWeightsEnc2, "hiddenEnc2");
	matrix_save(net->hiddenWeightsMu, "hiddenMu");
	matrix_save(net->hiddenWeightsDec, "hiddenDec");
	matrix_save(net->hiddenWeightsDec2, "hiddenDec2");
	matrix_save(net->outputWeights, "output");
	matrix_save(net->hiddenBiasEnc, "encBias");
	matrix_save(net->hiddenBiasEnc2, "enc2Bias");
	matrix_save(net->hiddenBiasMu, "muBias");
	matrix_save(net->hiddenBiasDec, "decBias");
	matrix_save(net->hiddenBiasDec2, "dec2Bias");
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
	net->hiddenEnc2 = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->mu = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hiddenDec = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hiddenDec2 = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hiddenWeightsEnc = matrix_load("hiddenEnc");
	net->hiddenWeightsEnc2 = matrix_load("hiddenEnc2");
	net->hiddenWeightsMu = matrix_load("hiddenMu");
	net->hiddenWeightsDec = matrix_load("hiddenDec");
	net->hiddenWeightsDec2 = matrix_load("hiddenDec2");
	net->outputWeights = matrix_load("output");

	net->hiddenBiasEnc = matrix_load("encBias");
	net->hiddenBiasEnc2 = matrix_load("enc2Bias");
	net->hiddenBiasMu = matrix_load("muBias");
	net->hiddenBiasDec = matrix_load("decBias");
	net->hiddenBiasDec2 = matrix_load("dec2Bias");
	net->outputBias = matrix_load("outputBias");

	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
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