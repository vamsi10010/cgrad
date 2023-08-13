#ifndef __ANN_H__
#define __ANN_H__

#include "grad.h"
#include "layer.h"

typedef struct ann_struct {
    int n_layers;
    LAYER **layers;
} ANN;

ANN *ann(int, int *, int);
VALUE **ann_forward(ANN *, VALUE **);
void ann_descend(ANN *, double, bool);

#endif // __ANN_H__