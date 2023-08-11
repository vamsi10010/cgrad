#ifndef __GRAD_H__
#define __GRAD_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum operation_enum {
    ADD,
    MUL,
    POW,
    RELU,
    TANH
} OPERATION; 

typedef struct value_struct {
    double val;
    double grad;
    bool requires_grad;
    void (* backward)(struct value_struct *);
    OPERATION op;
    struct value_struct *left;
    struct value_struct *right;
} VALUE;

typedef struct node_struct {
    VALUE *value;
    struct node_struct *prev;
} NODE;

// Operations

VALUE *add(VALUE *, VALUE *);
VALUE *mul(VALUE *, VALUE *);
VALUE *pow(VALUE *, VALUE *);
VALUE *relu(VALUE *);
VALUE *tanh(VALUE *);
VALUE *sub(VALUE *, VALUE *);
VALUE *div(VALUE *, VALUE *);
VALUE *neg(VALUE *);

// Backward Functions

void add_backward(VALUE *);
void mul_backward(VALUE *);
void pow_backward(VALUE *);
void relu_backward(VALUE *);
void tanh_backward(VALUE *);

void backward(VALUE *); // Backward pass





#endif // __GRAD_H__