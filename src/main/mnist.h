#ifndef __MNIST_H__
#define __MNIST_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "../nn/ann.h"
#include "../load/data.h"

#define NUM_LAYERS 3
#define LEARNING_RATE 0.1
#define EPOCHS 10
#define MOM_COEFF 0.9
#define REG_COEFF 0.1
#define BATCH_SIZE 64
#define OUTPUT_SIZE 10

int *perm(int);
void train(ANN *);

#endif // __MNIST_H__