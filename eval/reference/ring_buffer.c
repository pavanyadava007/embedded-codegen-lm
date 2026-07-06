void rb_init(ring_buffer_t *rb)
{
    rb->head = 0u;
    rb->tail = 0u;
}

bool rb_push(ring_buffer_t *rb, uint8_t byte)
{
    bool ok = false;
    uint32_t next = (rb->head + 1u) % RB_SIZE;
    if (next != rb->tail) {
        rb->buf[rb->head] = byte;
        rb->head = next;
        ok = true;
    }
    return ok;
}

bool rb_pop(ring_buffer_t *rb, uint8_t *out)
{
    bool ok = false;
    if (rb->tail != rb->head) {
        *out = rb->buf[rb->tail];
        rb->tail = (rb->tail + 1u) % RB_SIZE;
        ok = true;
    }
    return ok;
}
