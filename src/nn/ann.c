#include "ann.h"

ANN *ann(int num_layers, int *layer_sizes, OPERATION *activations, int num_inputs) {
    assert(num_layers > 0);
    assert(layer_sizes != NULL);
    assert(num_inputs > 0);

    ANN *a = malloc(sizeof(ANN));
    assert(a != NULL);

    a->n_layers = num_layers;

    a->layers = malloc(sizeof(LAYER *) * num_layers);
    assert(a->layers != NULL);

    for (int i = 0; i < num_layers; i++) {
        a->layers[i] = layer((i ? layer_sizes[i - 1] : num_inputs), layer_sizes[i], activations[i]);
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

VALUE *regularization(ANN *a, REG reg, double c) {
    assert(a != NULL);
    assert(c >= 0);

    VALUE *out = constant(0);

    for (int i = 0; i < a->n_layers; i++) {
        out = add(out, layer_regularization(a->layers[i], reg, c));
    }

    return out;
}

void ann_descend(ANN *a, double lr, bool momentum) {
    assert(a != NULL);
    assert(lr > 0);

    for (int i = 0; i < a->n_layers; i++) {
        layer_descend(a->layers[i], lr, momentum);
    }
}

void free_ann(ANN *a) {
    assert(a != NULL);

    for (int i = 0; i < a->n_layers; i++) {
        free_layer(a->layers[i]);
    }

    free(a->layers);
    free(a);
    a = NULL;
}

void zero_grad(ANN *a) {
    assert(a != NULL);

    for (int i = 0; i < a->n_layers; i++) {
        layer_zero_grad(a->layers[i]);
    }
}

VALUE *loss_fn(VALUE **yhat, double y, LOSS loss, int size) {
    assert(yhat != NULL);
    assert(size > 0);

    VALUE *out = constant(0);

    switch (loss) {
    case MSE:
        for (int i = 0; i < size; i++) {
            out = add(out, power(sub(yhat[i], constant(i == y)), constant(2)));
        }
        out = divide(out, constant(size));
        break;
    case CROSS_ENTROPY:
        for (int i = 0; i < size; i++) {
            out = add(out, mul(constant(i == (int) y), lg(yhat[i])));
        }
        out = neg(out);
        break;
    default:
        assert(false);
    }

    return out;
}


// Load and save functions