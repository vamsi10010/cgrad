#ifndef __NEURON_H__
#define __NEURON_H__

#include "../utils/grad.h"
#include "../utils/normal.h"

typedef enum regularization_enum {
    L1,
    L2
} REG;

typedef struct neuron_struct {
    int num_inputs;
    PARAM *params;
    VALUE **weights;
} NEURON;

NEURON *neuron(int);
VALUE *neuron_forward(NEURON *, VALUE **, OPERATION);
VALUE *neuron_regularization(NEURON *, REG, double);
void neuron_descend(NEURON *, double, bool);
void free_neuron(NEURON *);
void neuron_zero_grad(NEURON *);
void copy_weights(NEURON *);

#endif // __NEURON_H__