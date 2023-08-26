/** @file normal.c
 *  @brief Function for generating normally distributed random numbers.
 * 
 *  This contains function prototype for
 *  generating normally distributed random numbers.
 *
 *  @author Vamsi Deeduvanu (vamsi10010)
 */

# include "normal.h"

#define PI (3.141592653589793)

double normal(double mu, double sigma) {
  double r1 = 0.0;
  double r2 = 0.0;
  double x;

  do {r1 = drand48();} while (r1 == 0.0);
  do {r2 = drand48();} while (r2 == 0.0);
  x = sqrt(- 2.0 * log (r1)) * cos (2.0 * PI * r2);

  double value;
  value = mu + sigma * x;

  return value;
}
