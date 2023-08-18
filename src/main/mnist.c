#include "mnist.h"

int main() {
    int layer_sizes[NUM_LAYERS] = {100, 100, 10};

    OPERATION activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};
}