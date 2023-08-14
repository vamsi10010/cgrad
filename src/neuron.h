#ifndef __NEURON_H__
#define __NEURON_H__

#include "grad.h"
#include "normal.h"

typedef struct neuron_struct {
    int num_inputs;
    PARAM *params;
    VALUE *bias;
    VALUE **weights;
} NEURON;

NEURON *neuron(int);
VALUE *neuron_forward(NEURON *, VALUE **, OPERATION);
void neuron_descend(NEURON *, double, bool);
void free_neuron(NEURON *);

#endif // __NEURON_H__