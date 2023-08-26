/** @file normal.h
 *  @brief Function prototypes for generating normally distributed random numbers.
 * 
 *  This contains function prototypes for
 *  generating normally distributed random numbers.
 *  The function is implemented in normal.c.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

#ifndef __NORMAL_H__
#define __NORMAL_H__

# include <complex.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

double drand48(void);

/** 
 * @brief Returns a normally distributed random value with mean mu and standard
 * deviation sigma. 
 * 
 * This function uses an algortihm called [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform).
 * Use `normal(0, 1)` to generate a standard normal random variable.
 * 
 * @param mu The mean of the normal distribution.
 * @param sigma The standard deviation of the normal distribution.
*/
double normal(double mu, double sigma);

#endif // __NORMAL_H__