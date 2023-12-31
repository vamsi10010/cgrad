/** @file mnist.c
 *  @brief Trains and tests a feedforward neural network on the MNIST dataset.
 *  
 *  This contains the implementation for the functions used to train and test a
 *  feedforward neural network on the MNIST dataset.
 * 
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

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
    int *val_idx = idx + (int) (TRAIN_SIZE * 0.8);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    // train

    for (int i = 0; i < EPOCHS; i++) {
        printf("Epoch %2d ", i + 1);

        int *train_idx = idx;

        for (int j = 0; j < num_batches; j++) {
            if (j % 10 == 0) {
                printf("#");
                fflush(stdout);
            }

            // batch loss

            VALUE *loss = constant(0);

            for (int k = 0; k < BATCH_SIZE; k++) {

                VALUE **x = value_array(train_images[train_idx[k]], PIXELS);

                // normalize

                for (int l = 0; l < PIXELS; l++) {
                    x[l] = divide(x[l], constant(255));
                }

                // forward

                x = ann_forward(nn, x);

                // loss

                loss = add(loss, loss_fn(x, train_labels[train_idx[k]], CROSS_ENTROPY, OUTPUT_SIZE));

                free(x);
            }

            loss = divide(loss, constant(BATCH_SIZE));

            // regularization

            // loss = add(loss, regularization(nn, L2, REG_COEFF / (2.0 * BATCH_SIZE)));

            // backward

            backward(loss);
            free_values(&loss);

            // descend

            ann_descend(nn, LEARNING_RATE, true);

            zero_grad(nn);

            train_idx += BATCH_SIZE;
        }

        // validation

        printf("\n\tValidation ");

        double val_loss = 0;

        for (int j = 0; j < TRAIN_SIZE * 0.2; j++) {
            double *x = malloc(sizeof(double) * PIXELS);
            assert(x != NULL);

            x = memcpy(x, train_images[val_idx[j]], sizeof(double) * PIXELS);

            // normalize

            for (int k = 0; k < PIXELS; k++) {
                x[k] /= 255.0;
            }

            // forward

            x = ann_nograd_forward(nn, x);

            val_loss += loss_fn_nograd(x, train_labels[val_idx[j]], CROSS_ENTROPY, OUTPUT_SIZE);

            // free(x);
        }

        val_loss /= TRAIN_SIZE * 0.2;

        // output

        printf("\n\tLoss: %.5lf\n", val_loss);
        fflush(stdout);
    }

    // frees

    free_images(train_images, train_labels, TRAIN_SIZE);
    free(idx);
}

void test(ANN *nn) {
    assert(nn != NULL);

    // load testing data

    double **test_images;
    int *test_labels;

    read_csv(TEST_IMAGES, &test_images, &test_labels, TEST_SIZE);

    // test

    printf("Testing\n");

    int correct = 0;

    for (int i = 0; i < TEST_SIZE; i++) {
        double *x = malloc(sizeof(double) * PIXELS);
        assert(x != NULL);

        x = memcpy(x, test_images[i], sizeof(double) * PIXELS);

        // normalize

        for (int j = 0; j < PIXELS; j++) {
            x[j] /= 255.0;
        }

        // forward

        int pred = predict(nn, x, OUTPUT_SIZE);

        correct += (pred == test_labels[i]);

        // free(x);
    }

    // output

    printf("Accuracy: %.2lf%%\n", (100.0 * correct) / TEST_SIZE);
    fflush(stdout);

    // frees

    free_images(test_images, test_labels, TEST_SIZE);
}

int main() {
    int layer_sizes[NUM_LAYERS] = {36, 18, 10};
    OPERATION activations[NUM_LAYERS] = {RELU, RELU, SOFTMAX};

    ANN *nn = ann(NUM_LAYERS, layer_sizes, activations, PIXELS);

    train(nn);

    test(nn);

    free_ann(nn);
}