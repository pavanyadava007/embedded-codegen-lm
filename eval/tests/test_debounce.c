#include <assert.h>
#include <stdio.h>
#include "solution.h"

int main(void) {
    debounce_t d;
    debounce_init(&d, 0u);
    for (uint8_t i = 0u; i < (DEBOUNCE_COUNT - 1u); i++) {
        assert(debounce_update(&d, 1u) == 0u);     /* short glitch rejected */
    }
    assert(debounce_update(&d, 0u) == 0u);
    uint8_t s = 0u;
    for (uint8_t i = 0u; i < (DEBOUNCE_COUNT + 1u); i++) { s = debounce_update(&d, 1u); }
    assert(s == 1u);                               /* sustained press registers */
    printf("PASS\n");
    return 0;
}
