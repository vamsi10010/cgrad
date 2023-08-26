# cgrad
cgrad is a minimal backpropagation engine over a DAG, PyTorch style, and a simple neural network library built on top of it. It is written completely in C. cgrad aims to be memory efficient during training process by dynamically allocating memory for values and freeing them immediately after use. 

The file `mnist.c` contains a feedforward neural network built with cgrad. In its present configuration (2 hidden layers and 3 epochs), it achieves an accuracy of 96% on the MNIST dataset using less than 2GB of memory at any time of the training process. The data was obtained from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

To compile and run mnist.c, run the following in the directory `cgrad`:
```
make
make run
```
To run the cmocka tests,
```
make tests
```

cgrad is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd.git). His [video](https://youtu.be/VMj-3S1tku0?si=CDbpLHqpThtHuv9q) on micrograd is a fantastic resource for understanding backpropagation and neural networks.

Feel free to contribute to this project! I am new to deep learning and would love your feedback. Moving forward, I wish to incorporate CUDA support and optimize the overall training process further.



