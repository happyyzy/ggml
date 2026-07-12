#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

REQD_SUBGROUP_SIZE_64
kernel void kernel_fused_qknorm_rope_f32(
        global const char * src0,
        ulong               offset0,
        global const char * scale,
        ulong               offset_scale,
        global const char * pe,
        ulong               offset_pe,
        global       char * dst,
        ulong               offsetd,
        int                 d_head,
        int                 n_head,
        int                 n_tokens,
        int                 batch,
        ulong               nb01,
        ulong               nb02,
        ulong               nb03,
        ulong               scale_nb0,
        ulong               pe_nb1,
        ulong               pe_nb2,
        ulong               pe_nb3,
        ulong               dst_nb1,
        ulong               dst_nb2,
        ulong               dst_nb3,
        float               eps,
        local float       * sums) {
    src0  += offset0;
    scale += offset_scale;
    pe    += offset_pe;
    dst   += offsetd;

    const int row = (int)get_group_id(0);
    const int head = row % n_head;
    const int batch_id = row / n_head;
    const int token = (int)get_group_id(1);

    global const float * x = (global const float *)(src0 + (ulong)batch_id*nb03 + (ulong)token*nb02 + (ulong)head*nb01);

    float sum = 0.0f;
    for (int d = (int)get_local_id(0); d < d_head; d += (int)get_local_size(0)) {
        const float v = x[d];
        sum += v * v;
    }

    sum = sub_group_reduce_add(sum);
    if (get_sub_group_local_id() == 0) {
        sums[get_sub_group_id()] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_sub_group_id() == 0) {
        float total = 0.0f;
        const int n_subgroups = (int)(get_local_size(0) / get_max_sub_group_size());
        for (int i = (int)get_sub_group_local_id(); i < n_subgroups; i += (int)get_max_sub_group_size()) {
            total += sums[i];
        }
        total = sub_group_reduce_add(total);
        if (get_sub_group_local_id() == 0) {
            sums[0] = total;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float inv_rms = rsqrt(sums[0] / (float)d_head + eps);
    const int half_d = d_head / 2;
    const int out_head = batch_id * n_head + head;

    for (int pair = (int)get_local_id(0); pair < half_d; pair += (int)get_local_size(0)) {
        const int d0 = 2 * pair;
        const int d1 = d0 + 1;

        const float x0 = x[d0] * inv_rms * *((global const float *)(scale + (ulong)d0*scale_nb0));
        const float x1 = x[d1] * inv_rms * *((global const float *)(scale + (ulong)d1*scale_nb0));

        global const float * pe_base = (global const float *)(pe + (ulong)token*pe_nb3 + (ulong)pair*pe_nb2);
        const float p00 = pe_base[0]; // cos
        const float p10 = pe_base[1]; // sin
        global const float * pe_row1 = (global const float *)((global const char *)pe_base + pe_nb1);
        const float p01 = pe_row1[0]; // -sin
        const float p11 = pe_row1[1]; // cos

        global char * dst_base = dst + (ulong)out_head*dst_nb3 + (ulong)token*dst_nb2 + (ulong)pair*dst_nb1;
        *((global float *)(dst_base + 0*sizeof(float))) = x0*p00 + x1*p10;
        *((global float *)(dst_base + 1*sizeof(float))) = x0*p01 + x1*p11;
    }
}
