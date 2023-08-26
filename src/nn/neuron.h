/** @file neuron.h
 *  @brief Header file for neuron.c.
 * 
 *  This contains function prototypes and struct definitions
 *  to create neurons and perform operations on them.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#ifndef __NEURON_H__
#define __NEURON_H__

#include "../utils/grad.h"
#include "../utils/normal.h"

/**
 * @brief Different regularization techniques
 */
typedef enum regularization_enum {
    L1,
    L2
} REG;

/**
 * @brief A neuron in a neural network.
 */
typedef struct neuron_struct {
    int num_inputs;
    PARAM *params;
    VALUE **weights;
} NEURON;

/**
 * @brief Creates a neuron with the given number of inputs.
 * @param num_inputs The number of inputs to the neuron.
 * @return A pointer to the neuron.
 */
NEURON *neuron(int num_inputs);

/**
 * @brief Performs a forward pass on the neuron.
 * @param n The neuron.
 * @param x The inputs to the neuron.
 * @param activation The activation function to use.
 * @return The output of the neuron.
 */
VALUE *neuron_forward(NEURON *n, VALUE **x, OPERATION activation);

/**
 * @brief Performs a forward pass on the neuron without
 * building a computational graph.
 * @param n The neuron.
 * @param x The inputs to the neuron.
 * @param activation The activation function to use.
 * @return The output of the neuron.
 */
double neuron_nograd_forward(NEURON *n, double *x, OPERATION activation);

/**
 * @brief Calculates the regularization term for the neuron.
 * @param n The neuron.
 * @param reg The regularization technique to use.
 * @param c The regularization coefficient.
*/
VALUE *neuron_regularization(NEURON *n, REG reg, double c);

/**
 * @brief Performs a gradient descent step on the neuron.
 * @param n The neuron.
 * @param lr The learning rate.
 * @param momentum Whether to use momentum in gradient descent.
*/
void neuron_descend(NEURON *n, double lr, bool momentum);

/**
 * @brief Frees the memory allocated to the neuron.
 * @param n The neuron.
*/
void free_neuron(NEURON *n);

/**
 * @brief Performs a forward pass on the neuron without
 * building a computational graph.
 * @param n The neuron.
 * @param x The inputs to the neuron.
 * @param activation The activation function to use.
*/
VALUE *neuron_forward(NEURON *n, VALUE **x, OPERATION activation);


/**
 * @brief Copies the weights and biases into value structs
 * to construct the computational graph.
 * @param n The neuron.
*/
void copy_weights(NEURON *n);

/**
 * @brief Zeroes out the gradients of the neuron.
 * @param n The neuron.
*/
void neuron_zero_grad(NEURON *n);

#endif // __NEURON_H__