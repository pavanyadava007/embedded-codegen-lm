#include <assert.h>
#include <stdio.h>
#include "solution.h"

int main(void) {
    ring_buffer_t rb;
    uint8_t out;
    rb_init(&rb);
    assert(!rb_pop(&rb, &out));            /* empty */
    assert(rb_push(&rb, 42u));
    assert(rb_pop(&rb, &out) && (out == 42u));
    /* fill to capacity (accept RB_SIZE or RB_SIZE-1 depending on scheme) */
    uint32_t pushed = 0u;
    while (rb_push(&rb, (uint8_t)pushed)) { pushed++; if (pushed > RB_SIZE) { break; } }
    assert((pushed >= (RB_SIZE - 1u)) && (pushed <= RB_SIZE));
    /* drain preserves FIFO order */
    for (uint32_t i = 0u; i < pushed; i++) {
        assert(rb_pop(&rb, &out));
        assert(out == (uint8_t)i);
    }
    assert(!rb_pop(&rb, &out));
    printf("PASS\n");
    return 0;
}
