#include <assert.h>
#include <stdio.h>
#include "solution.h"

int main(void) {
    mavg_t m;
    mavg_init(&m);
    assert(mavg_update(&m, 10) == 10);             /* single sample */
    assert(mavg_update(&m, 20) == 15);             /* (10+20)/2 */
    for (uint32_t i = 0u; i < MA_WINDOW; i++) { (void)mavg_update(&m, 8); }
    assert(mavg_update(&m, 8) == 8);               /* window fully 8s */
    printf("PASS\n");
    return 0;
}
