#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "../src/grad.h"

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
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(null_test_success),
        cmocka_unit_test(test_gradients),
        cmocka_unit_test(test_repeating_values)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}