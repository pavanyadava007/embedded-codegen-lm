#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(is_pow2(1u)==1u);assert(is_pow2(2u)==1u);assert(is_pow2(3u)==0u);assert(is_pow2(0u)==0u);assert(is_pow2(1024u)==1u);printf("PASS\n");return 0;}
