#include "mnist.h"

int *perm(int n) {
    assert(n > 0);

    int *out = malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        out[i] = i;
    } 

    for (int i = n - 1; i >= 0; i--) {
        int j = rand() % (i + 1);
        
        int temp = out[i];
        out[i] = out[j];
        out[j] = temp;
    }

    return out;
}

void train(ANN *nn) {
    assert(nn != NULL);

    // load training data

    double **train_images;
    int *train_labels;

    read_csv(TRAIN_IMAGES, &train_images, &train_labels, TRAIN_SIZE);

    // shuffle dataset

    int *idx = perm(TRAIN_SIZE);
    int *train_idx = idx;
    int *val_idx = train_idx + (int) (TRAIN_SIZE * 0.2);

    // train

    for (int i = 0; i < EPOCHS; i++) {
        for (int j = 0; j < (TRAIN_SIZE / BATCH_SIZE); j++) {
            // get batch

            train_idx += BATCH_SIZE * j;

            // forward

            VALUE *loss = constant(0);

            for (int k = 0; k < BATCH_SIZE; k++) {
                VALUE **x = value_array(train_images[train_idx[k]], PIXELS);

                x = ann_forward(nn, x);

                // loss

                loss = add(loss, loss_fn(x, train_labels[train_idx[k]], CROSS_ENTROPY, OUTPUT_SIZE));

                free(x);
            }

            loss = divide(loss, constant(BATCH_SIZE));

            // regularization

            loss = add(loss, regularization(nn, L2, REG_COEFF / (2.0 * BATCH_SIZE)));

            // backward
            // descend
        }
    }



    // frees

    free_images(train_images, train_labels, TRAIN_SIZE);
    free(idx);
}

int main() {
    int layer_sizes[NUM_LAYERS] = {100, 100, 10};
    OPERATION activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};

    ANN *nn = ann(NUM_LAYERS, layer_sizes, activations, PIXELS);
}