#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main() {
	srand(time(NULL));
	int batchSize = 2;
	// TRAINING
	{
		int number_imgs = 50000, latent_size = 128, epochs = 1;
		double lr = 0.00005;
		Img** imgs = csv_to_imgs("data/mnist_train.csv", number_imgs);
		NeuralNetwork* net = aeCreate(latent_size, lr, batchSize);
		network_train_batch_imgs(net, imgs, number_imgs, batchSize, epochs, latent_size);
		network_save(net, "testing_net");
		imgs_free(imgs, number_imgs);
		free(imgs);
		network_free(net);
	}

	// TEST AND VISUALISE
	{
		int number_imgs = 10;
		int outputNum = 5;

		Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
		Img** outimgs = malloc(outputNum * sizeof(Img*));
		img_save(imgs[1]);
		img_save_new(imgs,number_imgs);
		NeuralNetwork* net = network_load("testing_net");
		outimgs[0] = network_predict(net, imgs[0], batchSize);
		outimgs[1] = network_predict(net, imgs[1],batchSize);
		outimgs[2] = network_predict(net, imgs[2],batchSize);
		outimgs[3] = network_predict(net, imgs[3],batchSize);
		outimgs[4] = network_predict(net, imgs[4],batchSize);
		
		img_save_new(outimgs,outputNum);

		imgs_free(imgs, number_imgs);
		free(imgs);
		imgs_free(outimgs, outputNum);
		free(outimgs);
		network_free(net);
	}
	return 0;
}
