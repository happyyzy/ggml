#include "htp-ctx.h"
#include "htp-tensor.h"

#include <hvx_hexagon_protos.h>
#include <string.h>

#include "hmx-w4a8-activation.h"

enum {
    TILE_ROWS = 64,
    TILE_COLS = 64,
    SLICE_BYTES = TILE_ROWS * TILE_COLS,
    BIAS_BYTES = 256,
    HMX_K_MAX = 2048,
};

struct w4a8_panel {
    const uint8_t *a;
    const uint8_t *b;
    uint8_t *digits;
    const uint32_t *bias;
    float *dst;
    const float *scales;
    uint32_t mt;
    uint32_t rows;
    uint32_t nt;
    uint32_t k;
    uint32_t ld;
    uint32_t nslices;
    const float *row_scales;
    const int32_t *correction;
};

static inline void read_slice(uint8_t *out, uint32_t bank) {
    const uint32_t select = (bank << 12) | 3; // retain accumulator, wrap to low byte
    asm volatile(
        "cvt.ub=acc(%2):sc0\n"
        "mxmem(%0,%3)=cvt\n"
        "cvt.ub=acc(%2):sc1\n"
        "mxmem(%1,%3)=cvt\n"
        :: "r"(out), "r"(out + SLICE_BYTES/2), "r"(select), "r"(0) : "memory");
}

static void multiply_panel(void *arg) {
    const struct w4a8_panel *p = arg;
    const size_t a_stride = 64 * p->k;
    const size_t b_stride = 32 * p->k;
    const size_t tile_bytes = SLICE_BYTES * p->nslices;
    uint8_t *out = p->digits;

#if __HVX_ARCH__ >= 75
    for (uint32_t s = 0; s < p->nslices; ++s) {
        asm volatile("bias=mxmem2(%0)" ::
                     "r"((uintptr_t)(p->bias + s*(BIAS_BYTES/4)) | s) : "memory");
    }
#endif
    for (uint32_t i = 0; i < p->mt; ++i) {
        for (uint32_t j = 0; j < p->nt; ++j) {
            const uint8_t *a = p->a + i*a_stride;
            const uint8_t *b = p->b + j*b_stride;
            asm volatile("mxclracc" ::: "memory");
            for (uint32_t kk = 0; kk < p->k; kk += HMX_K_MAX) {
                const uint32_t k = p->k - kk < HMX_K_MAX ? p->k - kk : HMX_K_MAX;
                asm volatile("{ activation.ub=mxmem(%0,%1):deep\n"
                             "weight.n=mxmem(%2,%3):2x }"
                             :: "r"(a + 64*kk), "r"(64*k-1),
                                "r"(b + 32*kk), "r"(32*k-1) : "memory");
            }
#if __HVX_ARCH__ < 75
            // v73 reads from one bias set, so load the scale for each byte slice.
            for (uint32_t s = 0; s < p->nslices; ++s) {
                asm volatile("bias=mxmem2(%0)" :: "r"(p->bias + s*(BIAS_BYTES/4)) : "memory");
                read_slice(out + s*SLICE_BYTES, 0);
            }
#else
            read_slice(out, 0);
            read_slice(out + SLICE_BYTES, 1);
            read_slice(out + 2*SLICE_BYTES, 2);
            if (p->nslices == 4) read_slice(out + 3*SLICE_BYTES, 3);
#endif
            out += tile_bytes;
        }
    }
    (void)*(volatile uint8_t *)(out-1);
}

static inline void store_scaled(float *dst, HVX_Vector value, HVX_Vector scale,
                                HVX_Vector row_scale, HVX_Vector correction) {
    value = Q6_Vw_vadd_VwVw(value, correction);
    scale = hvx_vec_mul_f32_f32(scale, row_scale);
    *(HVX_Vector *)dst = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(value), scale);
}

static void write_panel(unsigned int nth, unsigned int ith, void *arg) {
    const struct w4a8_panel *p = arg;
    const size_t tile_bytes = SLICE_BYTES * p->nslices;
    const uint32_t rows = (p->rows + 3) / 4;
    const uint32_t cols = p->nt * 2;
    const uint32_t row_begin = rows * ith / nth;
    const uint32_t row_end = rows * (ith + 1) / nth;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_Vector ones = Q6_V_vsplat_R(-1);
    for (uint32_t r = row_begin; r < row_end; ++r) {
        const uint32_t remaining = p->rows - 4*r;
        const uint8_t *tile = p->digits + (r/16)*p->nt*tile_bytes + (r%16)*128;
        HVX_Vector row_scale[4];
        for (unsigned i = 0; i < 4; ++i) {
            row_scale[i] = hvx_vec_splat_f32(p->row_scales[r*4 + i]);
        }
        for (uint32_t c = 0; c < cols; ++c) {
            const uint8_t *v = tile + (c/2)*tile_bytes + (c%2)*SLICE_BYTES/2;
            const HVX_Vector lo = *(const HVX_Vector *)v;
            const HVX_Vector mid = *(const HVX_Vector *)(v+SLICE_BYTES);
            const HVX_Vector hi = *(const HVX_Vector *)(v+2*SLICE_BYTES);
            HVX_Vector sign = hi;
            if (p->nslices == 4) sign = *(const HVX_Vector *)(v+3*SLICE_BYTES);
            const HVX_Vector s = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VbVb(zero, sign), ones, zero);
            const HVX_Vector top = p->nslices == 4
                ? Q6_V_vor_VV(Q6_Vub_vlsr_VubR(sign, 4), Q6_V_vand_VV(s, Q6_V_vsplat_R(0xf0f0f0f0)))
                : s;
            const HVX_VectorPair x = Q6_W_vshuff_VVR(mid, lo, -1);
            const HVX_VectorPair y = Q6_W_vshuff_VVR(top, hi, -1);
            const HVX_VectorPair uv = Q6_W_vshuff_VVR(Q6_V_lo_W(y), Q6_V_lo_W(x), -2);
            const HVX_VectorPair zw = Q6_W_vshuff_VVR(Q6_V_hi_W(y), Q6_V_hi_W(x), -2);
            // Deinterleave the four rows once their 32-bit words are assembled.
            const HVX_VectorPair left = Q6_W_vdeal_VVR(Q6_V_hi_W(uv), Q6_V_lo_W(uv), -4);
            const HVX_VectorPair right = Q6_W_vdeal_VVR(Q6_V_hi_W(zw), Q6_V_lo_W(zw), -4);
            const HVX_VectorPair even_rows = Q6_W_vdeal_VVR(Q6_V_lo_W(right), Q6_V_lo_W(left), -4);
            const HVX_VectorPair odd_rows = Q6_W_vdeal_VVR(Q6_V_hi_W(right), Q6_V_hi_W(left), -4);
            const HVX_Vector scale = *(const HVX_Vector *)(p->scales + c*32);
            const HVX_Vector correction = *(const HVX_Vector *)(p->correction + c*32);
            float *dst = p->dst + (size_t)r*4*p->ld + c*32;
            store_scaled(dst, Q6_V_lo_W(even_rows), scale, row_scale[0], correction);
            if (remaining > 1) {
                store_scaled(dst + p->ld, Q6_V_lo_W(odd_rows), scale, row_scale[1], correction);
            }
            if (remaining > 2) {
                store_scaled(dst + 2*p->ld, Q6_V_hi_W(even_rows), scale, row_scale[2], correction);
            }
            if (remaining > 3) {
                store_scaled(dst + 3*p->ld, Q6_V_hi_W(odd_rows), scale, row_scale[3], correction);
            }
        }
    }
}

int op_matmul_convrot(struct htp_ops_context *octx) {
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) return HTP_STATUS_OK;

    struct htp_context *ctx = octx->ctx;
    const struct htp_tensor *w = octx->src[0];
    const struct htp_tensor *x = octx->src[1];
    const struct htp_tensor *tail = octx->src[3];
    if (!ctx->hmx_queue || !(w->flags & HTP_TENSOR_REPACK)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    float *dst = htp_tensor_data(octx->dst);
    const float *activation = htp_tensor_data(x);
    const float *activation_tail = tail ? htp_tensor_data(tail) : NULL;
    const uint8_t *b = htp_tensor_data(w);
    const float *scale = htp_tensor_data(octx->src[2]);
    const uint32_t k = 2*w->ne[0], n = w->ne[1], k0 = x->ne[0];
    const int32_t *correction = (const int32_t *)(b + (size_t)k*n/2);
    const uint32_t m = x->ne[1] * x->ne[2] * x->ne[3];
    const uint32_t group_size = octx->op_params[0];

    work_queue_func_t prepare = group_size == 256
        ? prepare_w4a8_activations_h256 : prepare_w4a8_activations;
    const uint32_t nslices = k <= 4096 ? 3 : 4;
    const size_t tile_bytes = SLICE_BYTES*nslices;
    const size_t a_stride = 64*k;
    const size_t a_bytes_per_tile = a_stride + 64*sizeof(float);
    const size_t b_stride = 32*k;
    const size_t matrix_scales = sizeof(float)*n;
    const size_t fixed = hex_align_up(4*BIAS_BYTES + matrix_scales + sizeof(int32_t)*n, 2048);
    const uint32_t mt = (m+63)/64;
    const uint32_t nt = n/64;
    // Keep more activation rows resident for long K or narrow output panels.
    uint32_t nb = k >= 6144 || (m > 2048 && n <= 8192) ? 2 : 4;
    if (m >= 4096 && k <= 4096 && n >= 16384) {
        // Give wide ConvRot projections longer contiguous FP32 writes per row.
        nb = 8;
    } else if (m > 2048 && k < 4096 && n <= 8192 && nt%3 == 0) {
        nb = 3;
    }
    if (nb > nt) nb = nt;
    while (nb > 1 && fixed + a_bytes_per_tile + 2*nb*(b_stride+tile_bytes) > ctx->vtcm_size) nb /= 2;
    const size_t overhead = fixed + 2*nb*b_stride;
    const size_t row_tile_bytes = a_bytes_per_tile + 2*nb*tile_bytes;
    if (overhead + row_tile_bytes > ctx->vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    const uint32_t mb = hex_smin(mt, (ctx->vtcm_size - overhead)/row_tile_bytes);

    uint32_t *bias = (uint32_t *)ctx->vtcm_base;
    // Read bits 0..7, 8..15, 16..23 and (for long K) a signed byte at bit 20.
    const uint32_t shifts[4] = {20, 12, 4, 0};
    for (uint32_t s = 0; s < nslices; ++s) {
        for (uint32_t j = 0; j < 32; ++j) {
            bias[s*64+j] = shifts[s] << 10;
            bias[s*64+32+j] = 0;
        }
    }
    float *scales = (float *)(ctx->vtcm_base + 4*BIAS_BYTES);
    int32_t *col_correction = (int32_t *)(scales + n);
    memcpy(scales, scale, matrix_scales);
    memcpy(col_correction, correction, sizeof(int32_t)*n);

    uint8_t *a_buf = ctx->vtcm_base + fixed;
    uint8_t *b_buf[2] = {a_buf + mb*a_stride, a_buf + mb*a_stride + nb*b_stride};
    uint8_t *out_buf[2] = {b_buf[1] + nb*b_stride,
                           b_buf[1] + nb*b_stride + mb*nb*tile_bytes};
    float *row_scales = (float *)(out_buf[1] + mb*nb*tile_bytes);
    // The second weight buffer and both CVT buffers are idle during activation preparation.
    const size_t available = (uint8_t *)row_scales - b_buf[1];
    const uint32_t scratch_workers = available / (4*sizeof(float)*k);
    const uint32_t act_threads = scratch_workers < ctx->n_threads ? scratch_workers : ctx->n_threads;
    // Wide output panels with a partial activation tile favor the row pipeline.
    if (group_size == 256 && (nb <= 4 || m % 64 == 0) &&
        available >= (size_t)act_threads * 8*sizeof(float)*k) {
        prepare = prepare_w4a8_activations_h256_prefetch;
    }
    dma_queue_t dma = ctx->dma[0];
    struct w4a8_panel jobs[2];

    dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[0], b), nb*b_stride);
    for (uint32_t i = 0; i < mt; i += mb) {
        const uint32_t im = mt - i < mb ? mt-i : mb;
        const uint32_t rows = m - i*64 < im*64 ? m - i*64 : im*64;
        if (rows % 64 != 0) {
            memset(a_buf + (im-1)*a_stride, 128, a_stride);
        }
        const size_t first_row = (size_t)i * TILE_ROWS;
        struct w4a8_activations panel = {
            .src = activation + first_row*k0,
            .tail = activation_tail ? activation_tail + first_row*(k-k0) : NULL,
            .dst = a_buf, .scales = row_scales, .scratch = (HVX_Vector *)b_buf[1],
            .rows = rows, .k = k, .k0 = k0, .dma = ctx->dma, .group_size = group_size,
        };
        work_queue_run(ctx->work_queue, prepare, &panel, act_threads);
        dma_queue_flush(dma);
        uint32_t slot = 0;
        for (uint32_t j = 0; j < nt; j += nb, slot ^= 1) {
            jobs[slot] = (struct w4a8_panel){
                .a = a_buf, .b = b_buf[slot], .digits = out_buf[slot], .bias = bias,
                .dst = dst + (size_t)i*64*n + j*64, .scales = scales + j*64,
                .mt = im, .rows = rows, .nt = nt-j < nb ? nt-j : nb, .k = k, .ld = n, .nslices = nslices,
                .row_scales = row_scales,
                .correction = col_correction + j*64,
            };
            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(multiply_panel, &jobs[slot]));
            if (j+nb < nt) {
                const uint32_t next = nt-j-nb < nb ? nt-j-nb : nb;
                dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[slot^1], b + (j+nb)*b_stride), next*b_stride);
            }
            if (j) work_queue_run(ctx->work_queue, write_panel, &jobs[slot^1], ctx->n_threads);
            hmx_queue_pop(ctx->hmx_queue);
            dma_queue_flush(dma);
        }
        if (i+mb < mt) {
            dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[0], b), nb*b_stride);
        }
        work_queue_run(ctx->work_queue, write_panel, &jobs[slot^1], ctx->n_threads);
    }
    return HTP_STATUS_OK;
}
