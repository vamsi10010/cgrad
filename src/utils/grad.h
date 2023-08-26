/** @file grad.h
 *  @brief Function prototypes for grad.c
 * 
 *  This contains function prototypes for
 *  the backpropagation engine of cgrad.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#ifndef __GRAD_H__
#define __GRAD_H__

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 *  @brief Different operations
 *  supported by cgrad.
 */ 
typedef enum operation_enum {
    ADD,
    MUL,
    POW,
    MOD,
    EXP,
    LOG,
    RELU,
    TANH,
    SIGMOID,
    SOFTMAX,
    CONST,
    MAX
} OPERATION; 

/**
 *  @brief A parameter of a neuron in a neural network.
*/
typedef struct param_struct {
    double val;
    double grad;
    double momentum;
} PARAM;

/**
 *  @brief A singe unit/value in the computational graph
 *  during backpropagation.
 */
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

/**
 * @brief A node in a linked list.
 */

typedef struct node_struct {
    VALUE *value;
    struct node_struct *next;
} NODE;

// Operations

/**
 * @brief Adds two values.
 * @param a The first value.
 * @param b The second value.
 * @return The sum of the two values.
 */
VALUE *add(VALUE *a, VALUE *b);

/**
 * @brief Multiplies two values.
 * @param a The first value.
 * @param b The second value.
 * @return The product of the two values.
 */
VALUE *mul(VALUE *a, VALUE *b);

/**
 * @brief Raises a value to a power.
 * @param a The value.
 * @param b The power. Must be a constant value.
 * @return The value raised to the power.
 */
VALUE *power(VALUE *a, VALUE *b);

/**
 * @brief Takes the modulus of a value.
 * @param a The value.
 * @return The modulus of the value.
 */
VALUE *mod(VALUE *a);

/**
 * @brief Takes the exponential (e ^ a) of a value.
 * @param a The value.
 * @return The exponential of the value.
 */
VALUE *ex(VALUE *a);

/**
 * @brief Takes the natural logarithm of a value.
 * @param a The value.
 * @return The natural logarithm of the value.
 */
VALUE *lg(VALUE *a);

/**
 * @brief Takes the rectified linear unit of a value.
 * @param a The value.
 * @return The RELU of the value.
 */
VALUE *relu(VALUE *a);

/**
 * @brief Takes the hyperbolic tangent of a value.
 * @param a The value.
 * @return The tanh of the value.
 */
VALUE *tanhyp(VALUE *a);

/**
 * @brief Takes the sigmoid of a value.
 * @param a The value.
 * @return The sigmoid of the value.
 */
VALUE *sigmoid(VALUE *a);

/**
 * @brief Takes the difference of two values.
 * @param a The first value.
 * @param b The second value.
 * @return The difference of the two values.
 */
VALUE *sub(VALUE *a, VALUE *b);

/**
 * @brief Divides of two values.
 * @param a The numerator value.
 * @param b The denominator value.
 * @return The quotient of the two values.
 */
VALUE *divide(VALUE *a, VALUE *b);

/**
 * @brief Takes the negation (-a) of a value.
 * @param a The value.
 * @return The negation of the value.
 */
VALUE *neg(VALUE *a);

/**
 * @brief Applies softmax function to an array of values.
 * @param x The array of values.
 * @param size The size of the array.
 * @return A new array of values with softmax applied.
 */
VALUE **softmax(VALUE **x, int size);

/**
 * @brief Finds the maximum of two values.
 * @param a The first value.
 * @param b The second value.
 * @return The maximum of the two values.
 */
VALUE *max(VALUE *a, VALUE *b);

// Backward Functions

/**
 * @brief Backward pass for an addition operation.
 * @param v The value.
 */
void add_backward(VALUE *v);

/**
 * @brief Backward pass for a multiplication operation.
 * @param v The value.
 */
void mul_backward(VALUE *v);

/**
 * @brief Backward pass for a power operation.
 * @param v The value.
 */
void power_backward(VALUE *v);

/**
 * @brief Backward pass for a modulus operation.
 * @param v The value.
 */
void mod_backward(VALUE *v);

/**
 * @brief Backward pass for an exponential operation.
 * @param v The value.
 */
void ex_backward(VALUE *v);

/**
 * @brief Backward pass for a natural logarithm operation.
 * @param v The value.
 */
void lg_backward(VALUE *v);

/**
 * @brief Backward pass for a RELU operation.
 * @param v The value.
 */
void relu_backward(VALUE *v);

/**
 * @brief Backward pass for a tanh operation.
 * @param v The value.
 */
void tanh_backward(VALUE *v);

/**
 * @brief Backward pass for a sigmoid operation.
 * @param v The value.
 */
void sigmoid_backward(VALUE *v);

/**
 * @brief Backward pass over entire computational graph.
 * @param v The value.
 */
void backward(VALUE *v); 

// Helper Functions

/**
 * @brief Creates a value from a double.
 * @param val The input double.
 * @return The constant value.
 */
VALUE *constant(double);

/**
 * @brief Creates a value from a parameter.
 * @param val The input parameter.
 * @return The value.
 */
VALUE *parameter(PARAM *p);

/**
 * @brief Sorts a DAG of values into topological order.
 * @param v The head value.
 * @param head The head of the linked list.
 */
void build_topological_order(VALUE *v, NODE **head);

/**
 * @brief Builds a linked list node from a value.
 * @param v The value.
 * @return The node.
 */
NODE *build_node(VALUE *v);

/**
 * @brief Creates an array of values from an array of doubles.
 * @param arr The array of doubles.
 * @param size The size of the array.
 * @return The array of values.
 */
VALUE **value_array(double *arr, int size);

/**
 * @brief Finds the index of the maximum value in an array of values.
 * @param x The array of values.
 * @param size The size of the array.
 * @param out Pointer to output value.
 * @param idx Pointer to output index.
 * @return The index of the maximum value.
*/
void argmax(VALUE **x, int size, VALUE **out, int *idx);

// Graph Functions

/**
 * @brief Frees all values under input value in the graph.
 * @param v The input value.
*/
void free_values(VALUE **v);





#endif // __GRAD_H__