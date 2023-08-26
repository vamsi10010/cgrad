/** @file grad.c
 *  @brief Functions for backpropagation
 * 
 *  This contains the function implementations to perform
 *  bacpropagation on an equation. 
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#include "grad.h"   

VALUE *constant(double a) {
    VALUE *v = malloc(sizeof(VALUE));
    assert(v != NULL);

    *v = (VALUE) {
        .val = a,
        .grad = 0,
        .op = CONST,
        .left = NULL,
        .right = NULL
        };

    return v;
}

VALUE *parameter(PARAM *p) {
    assert(p != NULL);

    VALUE *v = malloc(sizeof(VALUE));
    assert(v != NULL);

    *v = (VALUE) {
        .val = p->val,
        .grad = p->grad,
        .op = CONST,
        .left = NULL,
        .right = NULL,
        .param = p
        };

    return v;
}

VALUE *add(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);

    VALUE *c = malloc(sizeof(VALUE));
    assert(c != NULL);  

    *c = (VALUE) {
        .val = a->val + b->val,
        .grad = 0,
        .backward = add_backward,
        .op = ADD,
        .left = a,
        .right = b
        };

    return c;
} 

void add_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad;
    v->right->grad += v->grad;
}

VALUE *mul(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);

    VALUE *c = malloc(sizeof(VALUE));
    assert(c != NULL);  

    *c = (VALUE) {
        .val = a->val * b->val,
        .grad = 0,
        .backward = mul_backward,
        .op = MUL,
        .left = a,
        .right = b
        };

    return c;
}

void mul_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * v->right->val;
    v->right->grad += v->grad * v->left->val;
}

VALUE *power(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(b->op == CONST);

    VALUE *c = malloc(sizeof(VALUE));
    assert(c != NULL);  

    *c = (VALUE) {
        .val = pow(a->val, b->val),
        .grad = 0,
        .backward = power_backward,
        .op = POW,
        .left = a,
        .right = b
        };

    return c;
}

void power_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * v->right->val * pow(v->left->val, v->right->val - 1);
}

VALUE *mod(VALUE *a) {
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = fabs(a->val),
        .grad = 0,
        .backward = mod_backward,
        .op = MOD,
        .left = a,
        .right = NULL
        };
    
    return b;
}

void mod_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * (v->left->val >= 0 ? 1 : -1);
}

VALUE *ex(VALUE *a) {
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = exp(a->val),
        .grad = 0,
        .backward = ex_backward,
        .op = EXP,
        .left = a,
        .right = NULL
        };

    return b;
}

void ex_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * v->val;
}

VALUE *lg(VALUE *a) {
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = log(a->val),
        .grad = 0,
        .backward = lg_backward,
        .op = LOG,
        .left = a,
        .right = NULL
        };

    return b;
}

void lg_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * (1 / v->left->val);
}

VALUE *relu(VALUE *a) {
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = a->val > 0 ? a->val : 0,
        .grad = 0,
        .backward = relu_backward,
        .op = RELU,
        .left = a,
        .right = NULL
        };

    return b;
}

void relu_backward(VALUE *v) {
    assert(v != NULL);

    v->left->grad += v->grad * (v->left->val > 0);
}

VALUE *tanhyp(VALUE *a) {
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = tanh(a->val),
        .grad = 0,
        .backward = tanh_backward,
        .op = TANH,
        .left = a,
        .right = NULL
        };

    return b;
}

void tanh_backward(VALUE *v) {
    // tanh'(x) = 1 - tanh(x)^2
    assert(v != NULL);

    v->left->grad += v->grad * (1 - pow(v->val, 2));
}

VALUE *sigmoid(VALUE *a) {
    // not actual sigmoid, this is fast sigmoid
    // f(x) = x / (1 + |x|)
    assert(a != NULL);

    VALUE *b = malloc(sizeof(VALUE));
    assert(b != NULL);

    *b = (VALUE) {
        .val = a->val / (1 + fabs(a->val)),
        .grad = 0,
        .backward = sigmoid_backward,
        .op = SIGMOID,
        .left = a,
        .right = NULL
        };

    return b;
}

void sigmoid_backward(VALUE *v) {
    // f'(x) = 1 / (1 + |x|)^2
    assert(v != NULL);

    v->left->grad += v->grad * 1 / pow(1 + fabs(v->left->val), 2);
}

VALUE *sub(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);

    return add(a, neg(b));
}

VALUE *divide(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);

    return mul(a, power(b, constant(-1)));
}

VALUE *neg(VALUE *a) {
    assert(a != NULL);

    return mul(a, constant(-1));
}

VALUE **softmax(VALUE **x, int size) {
    assert(x != NULL);
    assert(size > 0);

    VALUE **out = malloc(sizeof(VALUE *) * size);
    assert(out != NULL);

    VALUE *sum = constant(0);
    for (int i = 0; i < size; i++) {
        sum = add(sum, ex(x[i]));
    }

    for (int i = 0; i < size; i++) {
        out[i] = divide(ex(x[i]), sum);
    }

    free(x);

    return out;
}

void build_topological_order(VALUE *v, NODE **head) {
    assert(v != NULL);
    assert(head != NULL);

    if (!(v->visited)) {
        v->visited = true;

        if (v->left != NULL) {
            build_topological_order(v->left, head);
        }

        if (v->right != NULL) {
            build_topological_order(v->right, head);
        }

        NODE *node = malloc(sizeof(NODE));
        assert(node != NULL);

        *node = (NODE) {
            .value = v,
            .next = *head
            };

        *head = node;
    }
}

void backward(VALUE *v) {
    assert(v != NULL);

    NODE *head = NULL;

    build_topological_order(v, &head);

    v->grad = 1;
    while (head != NULL) {
        NODE *tmp = head;
        if (head->value->backward != NULL) (*(head->value->backward))(head->value);

        if (head->value->param != NULL) {
            head->value->param->grad += head->value->grad;
        }   

        head = head->next;
        tmp->value->visited = false;
        free(tmp);  
    }
}

void free_values(VALUE **v) {
    assert(v != NULL);
    
    NODE *head = NULL;

    build_topological_order(*v, &head);

    while (head != NULL) {
        NODE *tmp = head;
        head = head->next;
        free(tmp->value);
        free(tmp);
    }
}

VALUE **value_array(double *arr, int size) {
    assert(arr != NULL);
    assert(size > 0);

    VALUE **out = malloc(sizeof(VALUE *) * size);
    assert(out != NULL);

    for (int i = 0; i < size; i++) {
        out[i] = constant(arr[i]);
    }

    return out;
}

VALUE *max(VALUE *a, VALUE *b) {
    assert(a != NULL);
    assert(b != NULL);

    VALUE *c = malloc(sizeof(VALUE));
    assert(c != NULL);

    *c = (VALUE) {
        .val = a->val >= b->val ? a->val : b->val,
        .grad = 0,
        .backward = NULL,
        .op = MAX,
        .left = a,
        .right = b
        };

    return c;
}

void argmax(VALUE **args, int size, VALUE **out, int *idx) {
    assert(args != NULL);
    assert(size > 0);
    assert(out != NULL);
    assert(idx != NULL);

    *out = args[0];
    *idx = 0;

    for (int i = 1; i < size; i++) {
        *out = max(*out, args[i]);
        if (*out == args[i]) *idx = i;
    }
}








