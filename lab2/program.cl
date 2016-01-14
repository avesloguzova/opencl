__kernel void
do_reduce(__global float * out, int n, unsigned int offset)
{
    int thread_id = get_global_id(0);
    if (n / (offset * 2) > 256) {
        if (offset * (2 * thread_id + 2) - 1 < n) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            out[bi] += out[ai];
        }
    } else {
        while (n / (offset * 2) > 0) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            if (offset * (2 * thread_id + 2) - 1 < n) {
                int ai = offset * (2 * thread_id + 1) - 1;
                int bi = offset * (2 * thread_id + 2) - 1;
                out[bi] += out[ai];
            }
            offset *= 2;
        }
    }
}


__kernel void
do_sweep(__global float * out, int n, unsigned int offset)
{
    int thread_id = get_global_id(0);

    if (thread_id == 0 && offset == n/2)
        out[n-1] = 0;

    if (n / (offset * 2) > 256) {
        if (offset * (2 * thread_id + 2) - 1 < n) {
            int ai = offset * (2 * thread_id + 1) - 1;
            int bi = offset * (2 * thread_id + 2) - 1;
            float t = out[ai];
            out[ai] = out[bi];
            out[bi] += t;
        }
    } else {
        while (n / (offset * 2) <= 256) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            if (offset * (2 * thread_id + 2) - 1 < n) {
                int ai = offset * (2 * thread_id + 1) - 1;
                int bi = offset * (2 * thread_id + 2) - 1;
                float t = out[ai];
                out[ai] = out[bi];
                out[bi] += t;
            }
            offset /= 2;
        }
    }
}
