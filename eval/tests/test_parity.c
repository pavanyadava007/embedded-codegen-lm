#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(parity_odd(0u)==0u);assert(parity_odd(1u)==1u);assert(parity_odd(3u)==0u);assert(parity_odd(7u)==1u);printf("PASS\n");return 0;}
