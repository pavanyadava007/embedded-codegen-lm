#include <assert.h>
#include <stdio.h>
#include "solution.h"
int main(void){assert(reverse_bits8(0x01u)==0x80u);assert(reverse_bits8(0xFFu)==0xFFu);assert(reverse_bits8(0x00u)==0x00u);assert(reverse_bits8(0x0Fu)==0xF0u);printf("PASS\n");return 0;}
