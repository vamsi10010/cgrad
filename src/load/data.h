/** @file data.h
 *  @brief Function prototypes for data.c
 * 
 *  This contains the function prototypes for loading
 *  data from the MNIST dataset in CSV format.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

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

/**
 *  @brief Prints the image to the console
 *  @param image Array of doubles corresponding to pixels of flattened image
 *  @param size Number of pixels in the image
 *  @return void
*/
void print_image(double *image, int label);

/**
 *  @brief Reads a single image from the file
 *  @param fp File pointer to the CSV file with MNIST data
 *  @param size Pointer to the variable to store the label of the image
 *  @return Array of doubles corresponding to pixels of flattened image
*/
double *read_image(FILE *file, int *label);

/**
 *  @brief Reads the CSV file and stores the images and labels
 *  @param filename Name of the CSV file with MNIST data
 *  @param images Pointer to the 2D array to store flattened images
 *  @param labels Pointer to the array to store labels of images
 *  @param size Number of images in the CSV file
*/
void read_csv(char *path, double ***images_addr, int **labels_addr, int size);

/**
 *  @brief Frees the memory allocated to the images
 *  @param images Pointer to the 2D array with flattened images
 *  @param labels Pointer to the array with labels of images
 *  @param size Number of images 
*/
void free_images(double **, int *, int);

#endif // __DATA_H__