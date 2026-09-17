#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

__attribute__((qcom_reqd_sub_group_size("half")))
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_qknorm_rope_f32(
        global const char * src,
        ulong               src_offset,
        ulong               src_head_stride,
        ulong               src_token_stride,
        ulong               src_batch_stride,
        global const char * weight,
        ulong               weight_offset,
        global const char * theta,
        ulong               theta_offset,
        ulong               theta_token_stride,
        global char       * dst,
        ulong               dst_offset,
        int                 head_dim,
        int                 n_heads,
        int                 n_tokens,
        float               eps) {
    const int head  = get_group_id(0);
    const int token = get_group_id(1) % n_tokens;
    const int batch = get_group_id(1) / n_tokens;
    const int lane  = get_local_id(0);
    const int pairs = head_dim / 2;

    global const float * x = (global const float *) (src + src_offset +
        (ulong) head * src_head_stride + (ulong) token * src_token_stride +
        (ulong) batch * src_batch_stride);
    global const float * w = (global const float *) (weight + weight_offset);
    global const float * r = (global const float *) (theta + theta_offset +
        (ulong) token * theta_token_stride);

    const float2 x0 = lane < pairs ? vload2(lane, x) : (float2) (0.0f);
    float sum = dot(x0, x0);
    for (int p = lane + 64; p < pairs; p += 64) {
        const float2 xp = vload2(p, x);
        sum += dot(xp, xp);
    }
    const float inv_norm = rsqrt(sub_group_reduce_add(sum) / head_dim + eps);
    global float * out = (global float *) (dst + dst_offset) +
        ((ulong) batch * n_heads * n_tokens + (ulong) head * n_tokens + token) * head_dim;

    for (int p = lane; p < pairs; p += 64) {
        const float2 normalized = (p == lane ? x0 : vload2(p, x)) * inv_norm * vload2(p, w);
        const float4 rot = vload4(p, r);
        vstore2(normalized.x * rot.xz + normalized.y * rot.yw, p, out);
    }
}
