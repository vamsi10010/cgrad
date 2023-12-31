/** @file layer.c
 *  @brief Implementation for a layer of neurons in a feedforward neural network.
 *  
 *  This contains the implementation for the functions used to create and manipulate
 *  a layer of neurons in a feedforward neural network.
 * 
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#include "layer.h"

LAYER *layer(int num_inputs, int size, OPERATION activation) {
    assert(num_inputs > 0);
    assert(size > 0);
    assert(activation == RELU || activation == TANH || activation == SIGMOID ||
        activation == CONST || activation == SOFTMAX);

    LAYER *l = malloc(sizeof(LAYER));
    assert(l != NULL);

    l->size = size;
    l->activation = activation;

    l->neurons = malloc(sizeof(NEURON *) * size);
    assert(l->neurons != NULL);

    for (int i = 0; i < size; i++) {
        l->neurons[i] = neuron(num_inputs);
    }

    return l;
}

VALUE **layer_forward(LAYER *l, VALUE **x) {
    assert(l != NULL);
    assert(x != NULL);

    VALUE **out = malloc(sizeof(VALUE *) * l->size);
    assert(out != NULL);

    for (int i = 0; i < l->size; i++) {
        out[i] = neuron_forward(l->neurons[i], x, l->activation);
    }

    free(x);

    if (l->activation == SOFTMAX) {
        out = softmax(out, l->size);
    }

    return out;
}

double *layer_nograd_forward(LAYER *l, double *x) {
    assert(l != NULL);
    assert(x != NULL);

    double *out = malloc(sizeof(double) * l->size);
    assert(out != NULL);

    for (int i = 0; i < l->size; i++) {
        out[i] = neuron_nograd_forward(l->neurons[i], x, l->activation);
    }

    free(x);

    if (l->activation == SOFTMAX) {
        /* softmax */
        
        double denom = 0.0;
        for (int i = 0; i < l->size; i++) {
            denom += exp(out[i]);
        }

        for (int i = 0; i < l->size; i++) {
            out[i] = exp(out[i]) / denom;
        }
    }

    return out;
}

VALUE *layer_regularization(LAYER *l, REG reg, double c) {
    assert(l != NULL);
    assert(c >= 0);

    VALUE *out = constant(0);

    for (int i = 0; i < l->size; i++) {
        out = add(out, neuron_regularization(l->neurons[i], reg, c));
    }

    return out;
}

void layer_descend(LAYER *l, double lr, bool momentum) {
    assert(l != NULL);
    assert(lr > 0);

    for (int i = 0; i < l->size; i++) {
        neuron_descend(l->neurons[i], lr, momentum);
    }
}

void free_layer(LAYER *l) {
    assert(l != NULL);

    for (int i = 0; i < l->size; i++) {
        free_neuron(l->neurons[i]);
    }

    free(l->neurons);
    free(l);
    l = NULL;
}

void layer_zero_grad(LAYER *l) {
    assert(l != NULL);

    for (int i = 0; i < l->size; i++) {
        neuron_zero_grad(l->neurons[i]);
    }
}
