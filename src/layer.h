#ifndef __LAYER_H__
#define __LAYER_H__

#include "grad.h"
#include "neuron.h"

typedef struct layer_struct {
    int size;
    NEURON **neurons;
} LAYER;

LAYER *layer(int, int);
VALUE **layer_forward(LAYER *, VALUE **);
void layer_descend(LAYER *, double, bool);


#endif // __LAYER_H__