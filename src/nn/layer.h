/** @file layer.h
 *  @brief Header file for a layer of neurons in a feedforward neural network.
 *
 *  This contains the prototypes for the functions used to create and manipulate
 *  a layer of neurons in a feedforward neural network.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#ifndef __LAYER_H__
#define __LAYER_H__

#include "../utils/grad.h"
#include "neuron.h"

/**
 * @brief A layer of neurons in a feedforward neural network.
*/
typedef struct layer_struct {
    int size;
    NEURON **neurons;
    OPERATION activation;
} LAYER;

/**
 * @brief Creates a layer of neurons.
 * @param num_inputs The number of inputs to each neuron in the layer.
 * @param size The number of neurons in the layer.
 * @param activation The activation function for the layer.
 * @return A pointer to the layer.
*/
LAYER *layer(int num_inputs, int size, OPERATION activation);

/**
 * @brief Performs a forward pass on the layer.
 * @param l The layer.
 * @param x The input to the layer.
 * @return The output of the layer.
*/
VALUE **layer_forward(LAYER *l, VALUE **x);

/**
 * @brief Calculates the regularization term for the layer.
 * @param l The layer.
 * @param reg The regularization type.
 * @param c The regularization coefficient.
 * @return The regularization term.
*/
VALUE *layer_regularization(LAYER *l, REG reg, double c);

/**
 * @brief Performs a gradient descent step on the layer.
 * @param l The layer.
 * @param lr The learning rate.
 * @param momentum Whether to use momentum.
*/
void layer_descend(LAYER *l, double lr, bool momentum);

/**
 * @brief Frees the memory allocated to the layer.
 * @param l The layer.
*/
void free_layer(LAYER *l);

/**
 * @brief Sets the gradients of the layer to zero.
 * @param l The layer.
*/
void layer_zero_grad(LAYER *l);

/**
 * @brief Performs a forward pass on the layer without creating a graph.
 * @param l The layer.
 * @param x The input to the layer.
 * @return The output of the layer.
*/
double *layer_nograd_forward(LAYER *l, double *x);

#endif // __LAYER_H__