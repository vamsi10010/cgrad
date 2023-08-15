#ifndef __LAYER_H__
#define __LAYER_H__

#include "grad.h"
#include "neuron.h"

typedef struct layer_struct {
    int size;
    NEURON **neurons;
    OPERATION activation;
} LAYER;

LAYER *layer(int, int, OPERATION);
VALUE **layer_forward(LAYER *, VALUE **);
VALUE *layer_regularization(LAYER *, REG, double);
void layer_descend(LAYER *, double, bool);
void free_layer(LAYER *);
void layer_zero_grad(LAYER *);

#endif // __LAYER_H__