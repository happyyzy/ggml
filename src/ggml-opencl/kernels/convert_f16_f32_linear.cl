#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void kernel_f16_to_f32_linear(
        __global float * dst,
        const ulong      dst_offset,
        __global const half * src,
        const ulong      src_offset,
        const int        n) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    __global float * dst_ptr = (__global float *)((__global char *) dst + dst_offset);
    __global const half * src_ptr = (__global const half *)((__global const char *) src + src_offset);
    dst_ptr[i] = convert_float(src_ptr[i]);
}

__kernel void kernel_f32_copy_scale_linear(
        __global float * dst,
        const ulong      dst_offset,
        __global const float * src,
        const ulong      src_offset,
        const float      scale,
        const int        n) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    __global float * dst_ptr = (__global float *)((__global char *) dst + dst_offset);
    __global const float * src_ptr = (__global const float *)((__global const char *) src + src_offset);
    dst_ptr[i] = src_ptr[i] * scale;
}

// Copy + convert contiguous [D, N_src, H] (F16) to [D, N_dst, H] (F32),
// keeping the first N_dst tokens per head.
__kernel void kernel_f16_to_f32_nhd_crop(
        __global float * dst,
        const ulong      dst_offset,
        __global const half * src,
        const ulong      src_offset,
        const int        d,
        const int        n_head,
        const int        n_src,
        const int        n_dst,
        const int        src_start) {
    const int i = get_global_id(0);
    const int n = d * n_head * n_dst;
    if (i >= n) {
        return;
    }

    const int c = i % d;
    const int t = i / d;
    const int q = t % n_dst;
    const int h = t / n_dst;
    const int src_q = q + src_start;
    const int src_i = c + d * (src_q + n_src * h);

    __global float * dst_ptr = (__global float *)((__global char *) dst + dst_offset);
    __global const half * src_ptr = (__global const half *)((__global const char *) src + src_offset);
    dst_ptr[i] = convert_float(src_ptr[src_i]);
}

// Copy + convert contiguous [D, N_src, H] (F16) to [D, N_dst, H] (F32),
// preserving a head/tail token window:
// dst tokens = src[0:keep_head] + src[N_src-(N_dst-keep_head):N_src].
__kernel void kernel_f16_to_f32_nhd_keep_head_tail(
        __global float * dst,
        const ulong      dst_offset,
        __global const half * src,
        const ulong      src_offset,
        const int        d,
        const int        n_head,
        const int        n_src,
        const int        n_dst,
        const int        keep_head) {
    const int i = get_global_id(0);
    const int n = d * n_head * n_dst;
    if (i >= n) {
        return;
    }

    int kh = keep_head;
    if (kh < 0) {
        kh = 0;
    } else if (kh > n_dst) {
        kh = n_dst;
    }
    const int tail = n_dst - kh;

    const int c = i % d;
    const int t = i / d;
    const int q = t % n_dst;
    const int h = t / n_dst;

    int src_q = q;
    if (q >= kh) {
        src_q = (n_src - tail) + (q - kh);
    }

    const int src_i = c + d * (src_q + n_src * h);

    __global float * dst_ptr = (__global float *)((__global char *) dst + dst_offset);
    __global const half * src_ptr = (__global const half *)((__global const char *) src + src_offset);
    dst_ptr[i] = convert_float(src_ptr[src_i]);
}

// Reorder contiguous [H, Q, D] to contiguous [Q, H, D] for replay attention output.
__kernel void kernel_f32_reorder_hqd_to_qhd(
        __global float * dst,
        const ulong      dst_offset,
        __global const float * src,
        const ulong      src_offset,
        const int        h,
        const int        q,
        const int        d) {
    const int i = get_global_id(0);
    const int n = h * q * d;
    if (i >= n) {
        return;
    }

    const int di = i % d;
    const int t = i / d;
    const int qi = t % q;
    const int hi = t / q;

    // src is H-Q-D; dst is Q-H-D
    const int dst_i = di + d * (hi + h * qi);

    __global float * dst_ptr = (__global float *)((__global char *) dst + dst_offset);
    __global const float * src_ptr = (__global const float *)((__global const char *) src + src_offset);
    dst_ptr[dst_i] = src_ptr[i];
}
