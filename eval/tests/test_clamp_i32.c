#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(clamp_i32(5,0,10)==5);assert(clamp_i32(-3,0,10)==0);assert(clamp_i32(99,0,10)==10);printf("PASS\n");return 0;}
