void debounce_init(debounce_t *d, uint8_t initial_state)
{
    d->stable_state = initial_state;
    d->last_raw = initial_state;
    d->counter = 0u;
}

uint8_t debounce_update(debounce_t *d, uint8_t raw_sample)
{
    if (raw_sample == d->last_raw) {
        if (d->counter < DEBOUNCE_COUNT) {
            d->counter++;
        }
        if ((d->counter >= DEBOUNCE_COUNT) && (raw_sample != d->stable_state)) {
            d->stable_state = raw_sample;
        }
    } else {
        d->counter = 0u;
        d->last_raw = raw_sample;
    }
    return d->stable_state;
}
