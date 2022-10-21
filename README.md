### Autoencoder implementation
This project implements an autoencoder (AE) trained on the MNIST dataset. The AE uses no convolutional layers and the encoder and decoder parts are comprised of 2 dense layers each. 

The source code is written in Python using Pytorch/Numpy, Cython and C to compare execution times during training.

References: 
Basic C Neural Network
Normal distribution generator for C
PyTorch VAE

Current progress:
AE implemented using vanilla GD in C
AE implemented in PyTorch and Cython

Future aims: 
Implement the DEMON Adam optimiser to C source code.
Implement a disentangled VAE.
Test and compare execution times.