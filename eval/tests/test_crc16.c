#include <assert.h>
#include <stdio.h>
#include "solution.h"

int main(void) {
    const uint8_t check[] = {'1','2','3','4','5','6','7','8','9'};
    assert(crc16_ccitt(check, 9u) == 0x29B1u);   /* standard check value */
    assert(crc16_ccitt(check, 0u) == 0xFFFFu);   /* init value on empty  */
    const uint8_t a[] = {0x00u};
    assert(crc16_ccitt(a, 1u) == 0xE1F0u);
    printf("PASS\n");
    return 0;
}
