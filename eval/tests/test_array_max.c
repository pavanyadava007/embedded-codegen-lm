#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){int32_t a[]={3,7,2,9,1};assert(array_max(a,5u)==9);int32_t b[]={-5,-2,-9};assert(array_max(b,3u)==-2);printf("PASS\n");return 0;}
