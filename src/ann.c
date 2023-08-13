#include "ann.h"

ANN *ann(int num_layers, int *layer_sizes, int num_inputs) {
    assert(num_layers > 0);
    assert(layer_sizes != NULL);
    assert(num_inputs > 0);

    ANN *a = malloc(sizeof(ANN));
    assert(a != NULL);

    a->n_layers = num_layers;

    a->layers = malloc(sizeof(LAYER *) * num_layers);
    assert(a->layers != NULL);

    for (int i = 0; i < num_layers; i++) {
        a->layers[i] = layer(i ? layer_sizes[i - 1] : num_inputs, layer_sizes[i]);
    }

    return a;
}

VALUE **ann_forward(ANN *a, VALUE **x) {
    assert(a != NULL);
    assert(x != NULL);
    
    for (int i = 0; i < a->n_layers; i++) {
        x = layer_forward(a->layers[i], x);
    }

    return x;
}

void ann_descend(ANN *a, double lr, bool momentum) {
    assert(a != NULL);
    assert(lr > 0);

    for (int i = 0; i < a->n_layers; i++) {
        layer_descend(a->layers[i], lr, momentum);
    }
}

// TODO: Make a train, predict, and eval function
// Figure out how to free memory when predicting
// Make function to free ANN
// Load and save functions