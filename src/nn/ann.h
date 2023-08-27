/** @file ann.h
 *  @brief Header file for a feedforward neural network.
 *
 *  This contains the prototypes for the functions used to create 
 *  a feedforward neural network.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#ifndef __ANN_H__
#define __ANN_H__

#include "../utils/grad.h"
#include "layer.h"

/**
 * @brief Loss functions.
*/
typedef enum loss_enum {
    MSE,
    CROSS_ENTROPY
} LOSS;

/**
 * @brief A feedforward neural network.
*/
typedef struct ann_struct {
    int n_layers;
    LAYER **layers;
} ANN;

/**
 * @brief Creates a feedforward neural network.
 * @param num_layers The number of layers in the network.
 * @param layer_sizes The sizes of the layers in the network.
 * @param activations The activation functions for the layers in the network.
 * @param num_inputs The number of inputs to the network.
 * @return A pointer to the network.
*/
ANN *ann(int num_layers, int *layer_sizes, OPERATION *activations, int num_inputs);

/**
 * @brief Performs a forward pass on the network.
 * @param a The network.
 * @param x The input to the network.
 * @return The output of the network.
*/
VALUE **ann_forward(ANN *a, VALUE **x);

/**
 * @brief Calculates the regularization term.
 * @param a The network.
 * @param reg The regularization type.
 * @param c The regularization coefficient.
*/
VALUE *regularization(ANN *a, REG reg, double c);

/**
 * @brief Performs a gradient descent step on the network.
 * @param a The network.
 * @param lr The learning rate.
 * @param momentum Whether to use momentum.
*/
void ann_descend(ANN *a, double lr, bool momentum);

/**
 * @brief Frees the memory allocated to the network.
 * @param a The network.
*/
void free_ann(ANN *a);

/**
 * @brief Sets the gradients of the network to zero.
 * @param a The network.
*/
void zero_grad(ANN *a);

/**
 * @brief Calculates the loss of the network (classification only).
 * @param yhat The output of the network.
 * @param y The target class.
 * @param loss The loss function.
 * @param size The size of the output.
 * @return The loss.
*/
VALUE *loss_fn(VALUE **yhat, double y, LOSS loss, int size);

/**
 * @brief Performs a forward pass on the network without creating a graph.
 * @param a The network.
 * @param x The input to the network.
 * @return The output of the network.
*/
double *ann_nograd_forward(ANN *a, double *x);

/**
 * @brief Calculates the loss of the network (classification only) without values.
 * @param yhat The output of the network.
 * @param y The target class.
 * @param loss The loss function.
 * @param size The size of the output.
 * @return The loss.
*/
double loss_fn_nograd(double *yhat, double y, LOSS loss, int size);

/**
 * @brief Predicts the class of an input.
 * @param n The network.
 * @param x The input.
 * @param classes The number of classes.
 * @return The predicted class.
*/
int predict(ANN *n, double *x, int classes);

#endif // __ANN_H__