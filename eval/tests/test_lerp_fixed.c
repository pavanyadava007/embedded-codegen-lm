#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(lerp_q8(0,100,0u)==0);assert(lerp_q8(0,100,256u)==100);assert(lerp_q8(0,100,128u)==50);printf("PASS\n");return 0;}
