void mavg_init(mavg_t *m)
{
    for (uint32_t i = 0u; i < MA_WINDOW; i++) {
        m->buf[i] = 0;
    }
    m->idx = 0u;
    m->sum = 0;
    m->count = 0u;
}

int32_t mavg_update(mavg_t *m, int32_t sample)
{
    m->sum -= m->buf[m->idx];
    m->buf[m->idx] = sample;
    m->sum += sample;
    m->idx = (m->idx + 1u) % MA_WINDOW;
    if (m->count < MA_WINDOW) {
        m->count++;
    }
    return m->sum / (int32_t)m->count;
}
