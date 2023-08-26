/** @file data.c
 *  @brief Functions to load and free data from the MNIST dataset.
 *
 *  This contains the implementations for the functions used to read and 
 *  free data from the MNIST dataset in CSV format. It is used by the 
 *  neural network in `mnist.c`.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#include "data.h"

void print_image(double *image, int label) {
    assert(image != NULL);

    printf("Label: %d\n", label);

    for (int i = 0; i<PIXELS; i++) {
        printf("%3.0f", image[i]);
        if ((i+1) % 28 == 0) putchar('\n');
    }
}

double *read_image(FILE *file, int *label) {
    assert(file != NULL);

    // reading label

    if (fscanf(file, "%d", label) != 1) {
        printf("Error reading file\n");
        exit(1);
    }

    // reading image

    double *image = malloc(sizeof(double *) * PIXELS);
    assert(image != NULL);

    for (int i = 0; i < PIXELS; i++) {
        if (fscanf(file, ",%lf", image + i) != 1) {
            printf("Error reading file\n");
            exit(1);
        }
    }

    return image;
}

void read_csv(char *path, double ***images_addr, int **labels_addr, int size) {
    assert(path != NULL);
    assert(images_addr != NULL);
    assert(labels_addr != NULL);

    FILE *file = NULL;

    if ((file = fopen(path, "r")) == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    double **images = malloc(sizeof(double **) * size);
    assert(images != NULL);

    int *labels = malloc(sizeof(int *) * size);
    assert(labels != NULL);

    for (int i = 0; i < size; i++) {
        images[i] = read_image(file, labels + i);
    }

    *images_addr = images;
    *labels_addr = labels;

    fclose(file);
}

void free_images(double **images, int *labels, int size) {
    assert(images != NULL);
    assert(labels != NULL);

    for (int i = 0; i < size; i++) {
        free(images[i]);
    }

    free(images);
    free(labels);
}