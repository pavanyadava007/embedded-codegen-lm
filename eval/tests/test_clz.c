#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(clz32(0u)==32u);assert(clz32(1u)==31u);assert(clz32(0x80000000u)==0u);assert(clz32(0xFFu)==24u);printf("PASS\n");return 0;}
