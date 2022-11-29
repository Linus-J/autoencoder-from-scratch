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

	// TRAINING
	{
		int number_imgs = 50000, batchSize = 1, latent_size = 128;
		double lr = 0.00005;
		Img** imgs = csv_to_imgs("data/mnist_train.csv", number_imgs);
		NeuralNetwork* net = aeCreate(latent_size, lr, batchSize);
		network_train_batch_imgs(net, imgs, number_imgs);
		network_save(net, "testing_net");
		imgs_free(imgs, number_imgs);
		free(imgs);
		network_free(net);
	}

	// TEST AND VISUALISE
	{
		int number_imgs = 5;
		Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
		Img** outimgs = malloc(number_imgs * sizeof(Img*));
		img_save(imgs[1]);
		img_save_new(imgs,number_imgs);
		NeuralNetwork* net = network_load("testing_net");
		outimgs[0] = network_predict(net, imgs[0]);
		outimgs[1] = network_predict(net, imgs[1]);
		outimgs[2] = network_predict(net, imgs[2]);
		outimgs[3] = network_predict(net, imgs[3]);
		outimgs[4] = network_predict(net, imgs[4]);
		
		img_save_new(outimgs,number_imgs);

		imgs_free(imgs, number_imgs);
		free(imgs);
		imgs_free(outimgs, number_imgs);
		free(outimgs);
		network_free(net);
	}
	return 0;
}
