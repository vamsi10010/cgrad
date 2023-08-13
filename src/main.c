#include <stdio.h>

#include "normal.h"

int main() {
    for (int i = 0; i < 50; i++) {
        printf("%lf\n", normal(0.0, sqrt(2/10.0)));
    }
}