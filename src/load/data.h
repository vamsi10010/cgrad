#ifndef __DATA_H__
#define __DATA_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../utils/grad.h"

#define PIXELS 784
#define LABELS 10

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
#define TRIAL_SIZE 10

#define TRAIN_IMAGES "./data/mnist_train.csv"
#define TEST_IMAGES "./data/mnist_test.csv"
#define TRIAL_IMAGES "./data/mnist_trial.csv"

void print_image(double *, int);
double *read_image(FILE *, int *);
void read_csv(char *, double ***, int **, int);
void free_images(double **, int *, int);

#endif // __DATA_H__