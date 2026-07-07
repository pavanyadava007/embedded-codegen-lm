#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(map_range(5,0,10,0,100)==50);assert(map_range(0,0,10,0,100)==0);assert(map_range(10,0,10,0,100)==100);printf("PASS\n");return 0;}
