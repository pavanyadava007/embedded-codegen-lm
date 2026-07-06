void pid_init(pid_t *p, float kp, float ki, float kd, float dt,
              float out_min, float out_max)
{
    p->kp = kp;
    p->ki = ki;
    p->kd = kd;
    p->dt = dt;
    p->out_min = out_min;
    p->out_max = out_max;
    p->integ = 0.0f;
    p->prev_err = 0.0f;
}

float pid_update(pid_t *p, float setpoint, float measured)
{
    float err = setpoint - measured;
    float integ_new = p->integ + (err * p->dt);
    float deriv = (err - p->prev_err) / p->dt;
    float out = (p->kp * err) + (p->ki * integ_new) + (p->kd * deriv);

    if (out > p->out_max) {
        out = p->out_max;
    } else if (out < p->out_min) {
        out = p->out_min;
    } else {
        p->integ = integ_new;   /* clamped anti-windup: integrate only when unsaturated */
    }
    p->prev_err = err;
    return out;
}
