#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "../src/utils/grad.h"
#include "../src/nn/neuron.h"
#include "../src/nn/layer.h"
#include "../src/nn/ann.h"
#include "../src/load/data.h"

static void null_test_success(void **state) {
    (void) state; /* unused */
}

static void test_gradients(void **state) {
    VALUE *a = constant(2);
    VALUE *b = constant(3);
    VALUE *c = constant(4);
    VALUE *d = constant(5);

    VALUE *e = add(a, b);
    VALUE *f = mul(c, d);
    VALUE *g = add(e, f);

    backward(g);

    assert_int_equal(a->grad, 1);
    assert_int_equal(b->grad, 1);
    assert_int_equal(c->grad, 5);
    assert_int_equal(d->grad, 4);

    free_values(&g);
}

static void test_repeating_values(void **state) {
    VALUE *a = constant(2);
    VALUE *b = constant(3);
    VALUE *c = constant(4);

    VALUE *e = add(a, b);
    VALUE *f = mul(c, a);
    VALUE *g = add(e, f);

    backward(g);

    assert_int_equal(a->grad, 5);
    assert_int_equal(b->grad, 1);
    assert_int_equal(c->grad, 2);

    free_values(&g);
}

static void test_relu(void **state) {
    VALUE *a = constant(2);
    VALUE *b = constant(-3);

    VALUE *c = relu(a);
    VALUE *d = relu(b);

    backward(c);
    backward(d);

    assert_double_equal(c->val, 2, 0);
    assert_double_equal(d->val, 0, 0);
    assert_int_equal(a->grad, 1);
    assert_int_equal(b->grad, 0);

    free_values(&c);
    free_values(&d);
}

static void test_tanh(void **state) {
    VALUE *a = constant(log(2));

    VALUE *b = tanhyp(a);

    backward(b);

    assert_double_equal(b->val, tanh(a->val), 0.0001);
    assert_double_equal(a->grad, (1 - pow(tanh(a->val), 2)), 0.0001);

    free_values(&b);
}

static void test_sigmoid(void **state) {
    VALUE *a = constant(2);
    VALUE *b = constant(-3);

    VALUE *c = sigmoid(a);
    VALUE *d = sigmoid(b);

    backward(c);
    backward(d);

    assert_double_equal(c->val, a->val / (1 + fabs(a->val)), 0.0001);
    assert_double_equal(d->val, b->val / (1 + fabs(b->val)), 0.0001);

    assert_double_equal(a->grad, 1 / pow(1 + fabs(a->val), 2), 0.0001);
    assert_double_equal(b->grad, 1 / pow(1 + fabs(b->val), 2), 0.0001);

    free_values(&c);
    free_values(&d);
}

static void test_neuron(void **state) {
    // y = w1*a + w2*b + w3*c + b
    NEURON *n = neuron(3);

    VALUE *a = constant(2);
    VALUE *b = constant(3);
    VALUE *c = constant(4);

    VALUE *x[3] = {a, b, c};

    VALUE *y = neuron_forward(n, x, CONST);

    backward(y);

    // dy/dw1 = a
    assert_int_equal(n->params[1].grad, a->val);
    // dy/dw2 = b
    assert_int_equal(n->params[2].grad, b->val);
    // dy/dw3 = c
    assert_int_equal(n->params[3].grad, c->val);
    // dy/db = 1
    assert_int_equal(n->params[0].grad, 1);
    //dy/da = w1
    assert_double_equal(a->grad, n->params[1].val, 0.0001);
    //dy/db = w2
    assert_double_equal(b->grad, n->params[2].val, 0.0001);
    //dy/dc = w3
    assert_double_equal(c->grad, n->params[3].val, 0.0001);

    neuron_zero_grad(n);

    assert_int_equal(n->params[1].grad, 0);
    assert_int_equal(n->params[2].grad, 0);
    assert_int_equal(n->params[3].grad, 0);
    assert_int_equal(n->params[0].grad, 0);

    free_values(&y);

    free_neuron(n);
}

static void test_neuron_descend(void **state) {
    // y = w1*a + w2*b + w3*c + b
    NEURON *n = neuron(3);

    double w1 = n->params[1].val;
    double w2 = n->params[2].val;
    double w3 = n->params[3].val;
    double bi = n->params[0].val;

    VALUE *a = constant(2);
    VALUE *b = constant(3);
    VALUE *c = constant(4);

    VALUE *x[3] = {a, b, c};

    VALUE *y = neuron_forward(n, x, CONST);

    backward(y);

    neuron_descend(n, 0.1, false);

    assert_double_equal(n->params[1].val, w1 - 0.1 * a->val, 0.0001);
    assert_double_equal(n->params[2].val, w2 - 0.1 * b->val, 0.0001);
    assert_double_equal(n->params[3].val, w3 - 0.1 * c->val, 0.0001);
    assert_double_equal(n->params[0].val, bi - 0.1, 0.0001);

    neuron_zero_grad(n);

    free_values(&y);

    free_neuron(n);
}

static void test_dataloader(void **state) {
    double **images;
    int *labels;

    read_csv(TRIAL_IMAGES, &images, &labels, TRIAL_SIZE);

    int i =  9;
    double *image = images[i];
    int label = labels[i];

    assert_int_equal(label, 4);

    print_image(image, label);
    print_message("EYE TEST FOR 4\n");

    free_images(images, labels, TRIAL_SIZE);
}

static void test_ann(void **state) {
    int sizes[3] = {10, 10, 2};
    OPERATION ops[3] = {RELU, RELU, SIGMOID};

    ANN *nn = ann(3, sizes, ops, 3);

    VALUE **x = malloc(sizeof(VALUE *) * 3);

    VALUE *a = constant(2);
    VALUE *b = constant(3);
    VALUE *c = constant(4);

    double w1 = nn->layers[0]->neurons[0]->params[1].val;
    assert_double_equal(nn->layers[0]->neurons[0]->params[1].grad, 0, 0);

    x[0] = a;
    x[1] = b;
    x[2] = c;

    x = ann_forward(nn, x);

    VALUE **y = malloc(sizeof(VALUE *) * 2);

    VALUE *d = constant(1);
    VALUE *e = constant(3);

    y[0] = d;
    y[1] = e;

    VALUE *loss = constant(0);

    for (int i = 0; i < sizes[2]; i++) {
        loss = add(loss, power(sub(x[i], y[i]), constant(2)));
    }

    loss = divide(loss, constant(sizes[2]));

    loss = add(loss, regularization(nn, L2, 0.1));

    printf("LOSS: %f\n", loss->val);

    backward(loss);
    free_values(&loss);

    assert_double_not_equal(nn->layers[0]->neurons[0]->params[1].grad, 0, 0);
    ann_descend(nn, 0.1, false);
    assert_double_not_equal(nn->layers[0]->neurons[0]->params[1].val, w1, 0);
    zero_grad(nn);
    assert_double_equal(nn->layers[0]->neurons[0]->params[1].grad, 0, 0);

    free(x);
    free(y);

    free_ann(nn);

    // CHECK VALGRIND OUTPUT
}
    

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(null_test_success),
        cmocka_unit_test(test_gradients),
        cmocka_unit_test(test_repeating_values),
        cmocka_unit_test(test_relu),
        cmocka_unit_test(test_tanh),
        cmocka_unit_test(test_sigmoid),
        cmocka_unit_test(test_neuron),
        cmocka_unit_test(test_neuron_descend),
        cmocka_unit_test(test_dataloader),
        cmocka_unit_test(test_ann)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}