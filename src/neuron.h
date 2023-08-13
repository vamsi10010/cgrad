#ifndef __NEURON_H__
#define __NEURON_H__

#include "grad.h"
#include "normal.h"

typedef struct neuron_struct {
    VALUE *bias;
    VALUE **weights;
    PARAM *params;
} NEURON;

NEURON *neuron(int);
void free_neuron(NEURON *);
VALUE *neuron_forward(NEURON *, VALUE **);

#endif // __NEURON_H__