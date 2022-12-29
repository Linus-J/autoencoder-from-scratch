#### C autoencoder fork from C based MNIST classifier [here](https://github.com/markkraay/mnist-from-scratch)
#### Python autoencoder referenced from [here](https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1)

## Autoencoder implementation


This project implements an autoencoder (AE) trained on the MNIST dataset. The AE uses no convolutional layers and the encoder and decoder parts are comprised of 2 dense layers each. 

The source code is written in Python using Pytorch/Numpy, Cython and C to compare execution times during training.

### References: 
- [Basic Neural Network in C](https://github.com/markkraay/mnist-from-scratch "Basic Neural Network in C")
- [Normal distribution generator for C](https://people.sc.fsu.edu/~jburkardt/cpp_src/ziggurat_inline/ziggurat_inline.html "Normal distribution generator for C")
- [PyTorch AE](https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1 "PyTorch AE")
- [DEMON Adam optimiser](https://github.com/JRC1995/DemonRangerOptimizer "DEMON Adam optimiser")

### Current progress:
- AE implemented using vanilla SGD in C
- Implemented batch training and composite matrix operations in C
- AE implemented in PyTorch and Cython with DEMON Adam optimiser

### Future aims: 
- Implement DEMON Adam optimiser to C source code.
- Implement a disentangled VAE in all languages.
- Test and compare execution times.
