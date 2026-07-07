#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(sat_add_u8(100u,50u)==150u);assert(sat_add_u8(200u,100u)==255u);assert(sat_add_u8(255u,255u)==255u);printf("PASS\n");return 0;}
