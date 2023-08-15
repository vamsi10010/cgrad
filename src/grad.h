#ifndef __GRAD_H__
#define __GRAD_H__

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum operation_enum {
    ADD,
    MUL,
    POW,
    RELU,
    TANH,
    SIGMOID,
    CONST
} OPERATION; 

typedef struct param_struct {
    double val;
    double grad;
    double momentum;
} PARAM;

typedef struct value_struct {
    double val;
    double grad;
    void (*backward)(struct value_struct *);
    OPERATION op;
    struct value_struct *left;
    struct value_struct *right;
    bool visited;
    PARAM *param;
} VALUE;

typedef struct node_struct {
    VALUE *value;
    struct node_struct *next;
} NODE;

// Operations

VALUE *add(VALUE *, VALUE *);
VALUE *mul(VALUE *, VALUE *);
VALUE *power(VALUE *, VALUE *);
VALUE *relu(VALUE *);
VALUE *tanhyp(VALUE *);
VALUE *sigmoid(VALUE *);
VALUE *sub(VALUE *, VALUE *);
VALUE *divide(VALUE *, VALUE *);
VALUE *neg(VALUE *);

// Backward Functions

void add_backward(VALUE *);
void mul_backward(VALUE *);
void power_backward(VALUE *);
void relu_backward(VALUE *);
void tanh_backward(VALUE *);
void sigmoid_backward(VALUE *);

void backward(VALUE *); // Backward pass


// Helper Functions
VALUE *constant(double);
VALUE *parameter(PARAM *);
void build_topological_order(VALUE *, NODE **);
NODE *build_node(VALUE *);

// Graph Functions
void free_values(VALUE **);





#endif // __GRAD_H__