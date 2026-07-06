#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "solution.h"

int main(void) {
    pid_t p;
    pid_init(&p, 2.0f, 0.5f, 0.0f, 0.01f, -10.0f, 10.0f);
    float o = pid_update(&p, 1.0f, 0.0f);          /* P dominant */
    assert((o > 1.9f) && (o < 2.2f));
    pid_init(&p, 100.0f, 0.0f, 0.0f, 0.01f, -10.0f, 10.0f);
    o = pid_update(&p, 100.0f, 0.0f);              /* clamped at max */
    assert(fabsf(o - 10.0f) < 1e-5f);
    pid_init(&p, 0.0f, 1000.0f, 0.0f, 0.01f, -10.0f, 10.0f);
    for (int i = 0; i < 1000; i++) { (void)pid_update(&p, 100.0f, 0.0f); }
    o = pid_update(&p, 0.0f, 0.0f);                /* anti-windup: recovers */
    assert(o <= (10.0f + 1e-3f));
    printf("PASS\n");
    return 0;
}
