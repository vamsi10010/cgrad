/** @file mnist.h
 *  @brief Trains and tests a feedforward neural network on the MNIST dataset.
 *
 *  This contains the prototypes for the functions used to train and test a
 *  feedforward neural network on the MNIST dataset. 
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

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
#define EPOCHS 3
#define MOM_COEFF 0.9
#define REG_COEFF 0.1
#define BATCH_SIZE 32
#define OUTPUT_SIZE 10

/**
 *  @brief Generates a random permutation of the integers from 0 to n - 1.
 *  @param n The number of integers to permute.
 *  @return A pointer to the array of integers.
 */
int *perm(int n);

/**
 *  @brief Trains the neural network on the MNIST dataset.
 *  @param nn The neural network.
 */
void train(ANN *nn);

/**
 *  @brief Tests the neural network on the MNIST dataset.
 *  @param nn The neural network.
 */
void test(ANN *);

#endif // __MNIST_H__