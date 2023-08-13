#include "neuron.h" 

NEURON *neuron(int num_inputs) {
    NEURON *n = malloc(sizeof(NEURON));
    assert(n != NULL);
    assert(num_inputs > 0);

    n->num_inputs = num_inputs;

    n->params = malloc(sizeof(PARAM) * (num_inputs + 1));
    assert(n->params != NULL);

    /* initializing weights and biases with normal distribution */

    for (int i = 0; i < num_inputs + 1; i++) {
        n->params[i] = (PARAM) {
            .val = normal(0.0, sqrt(2.0 / num_inputs))
        };
    }

    n->bias = parameter(&(n->params[0]));

    n->weights = malloc(sizeof(VALUE *) * num_inputs);
    assert(n->weights != NULL);

    for (int i = 0; i < num_inputs; i++) {
        n->weights[i] = parameter(&(n->params[i + 1]));
    }

    return n;
}

VALUE *neuron_forward(NEURON *n, VALUE **x) {
    assert(n != NULL);
    assert(x != NULL);

    VALUE *out = constant(0);

    for (int i = 0; i < n->num_inputs; i++) {
        out = add(out, mul(n->weights[i], x[i]));
    }
    out = add(out, n->bias);

    return out;
}

void neuron_descend(NEURON *n, double lr, bool momentum) {
    assert(n != NULL);
    assert(lr > 0);

    double change;
    double beta = 0.9; // momentum coefficient
    for (int i = 0; i < n->num_inputs + 1; i++) {
        change = lr * n->params[i].grad;
        if (momentum) change += lr * beta * n->params[i].momentum;

        n->params[i].val -= change;
    }
}
