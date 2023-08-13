#include "neuron.h" 

NEURON *neuron(int num_inputs) {
    NEURON *n = malloc(sizeof(NEURON));
    assert(n != NULL);
    assert(num_inputs > 0);

    n->params = malloc(sizeof(PARAM) * (num_inputs + 1));
    assert(n->params != NULL);

    /* initializing weights and biases with normal distribution */

    for (int i = 0; i < num_inputs + 1; i++) {
        n->params[i] = (PARAM) {
            .val = normal(0.0, sqrt(2.0 / num_inputs))
        };
    }

    n->bias = constant(0);
    n->weights = malloc(sizeof(VALUE *) * num_inputs);
    assert(n->weights != NULL);

    for (int i = 0; i < num_inputs; i++) {
        n->weights[i] = constant(0);
    }

    return n;
}