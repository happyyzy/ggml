#include "htp-ctx.h"
#include "htp-tensor.h"

#include <hvx_hexagon_protos.h>
#include <string.h>

#include "hmx-w4a8-activation.h"

enum {
    TILE_ROWS   = 64,
    TILE_COLS   = 64,
    SLICE_BYTES = TILE_ROWS * TILE_COLS,
    BIAS_BYTES  = 256,
    HMX_K_MAX   = 1024,
};

struct w8a8_panel {
    const uint8_t* a;
    const uint8_t* b;
    uint8_t* digits;
    const uint32_t* bias;
    float* dst;
    const float* scales;
    uint32_t mt;
    uint32_t rows;
    uint32_t nt;
    uint32_t output_nt;
    uint32_t k;
    uint32_t ld;
    const float* row_scales;
    const int32_t* correction;
};

static inline void read_accumulator(uint8_t* out, const uint32_t* bias) {
    for (uint32_t s = 0; s < 4; ++s) {
#if __HVX_ARCH__ < 75
        asm volatile("bias=mxmem2(%0)" ::"r"(bias + s * (BIAS_BYTES / 4))
                     : "memory");
        const uint32_t select = 3;
#else
        (void)bias;
        const uint32_t select = (s << 12) | 3;
#endif
        // Unqualified CVT combines both nibble contributions of weight.b.
        asm volatile("cvt.ub=acc(%1)\n mxmem(%0,%2):cm=cvt" ::"r"(out + s * SLICE_BYTES), "r"(select), "r"(0)
                     : "memory");
    }
}

static void multiply_panel(void* arg) {
    const struct w8a8_panel* p = arg;
    const size_t stride        = 64 * p->k;
    const size_t tile_bytes    = 4 * SLICE_BYTES;
    uint8_t* out               = p->digits;
#if __HVX_ARCH__ >= 75
    for (uint32_t s = 0; s < 4; ++s) {
        asm volatile("bias=mxmem2(%0)" ::
                         "r"((uintptr_t)(p->bias + s * (BIAS_BYTES / 4)) | s)
                     : "memory");
    }
#endif
    for (uint32_t i = 0; i < p->mt; ++i) {
        for (uint32_t j = 0; j < p->nt; ++j) {
            const uint8_t* a = p->a + i * stride;
            const uint8_t* b = p->b + j * stride;
            asm volatile("mxclracc\n mxswapacc\n mxclracc\n mxswapacc" ::
                             : "memory");
            for (uint32_t kk = 0; kk < p->k; kk += HMX_K_MAX) {
                const uint32_t k     = hex_smin(p->k - kk, HMX_K_MAX);
                const uint32_t range = 64 * k - 1;
                asm volatile(
                    "{ activation.ub=mxmem(%0,%2):deep\n"
                    "weight.b=mxmem(%1,%2):deep }" ::"r"(a + 64 * kk),
                    "r"(b + 64 * kk), "r"(range)
                    : "memory");
            }
            read_accumulator(out, p->bias);
            asm volatile("mxswapacc" ::
                             : "memory");
            read_accumulator(out + SLICE_BYTES / 2, p->bias);
            asm volatile("mxswapacc" ::
                             : "memory");
            out += tile_bytes;
        }
        out += (p->output_nt - p->nt) * tile_bytes;
    }
    const size_t last = ((p->mt - 1) * p->output_nt + p->nt) * tile_bytes;
    (void)*(volatile uint8_t*)(p->digits + last - 1);
}

static inline void store_scaled(float* dst, HVX_Vector value, HVX_Vector scale, HVX_Vector row_scale, HVX_Vector correction) {
    value                   = Q6_Vw_vadd_VwVw(value, correction);
    scale                   = hvx_vec_mul_f32_f32(scale, row_scale);
    const HVX_Vector result = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(value), scale);
    asm volatile("vmem(%0+#0):nt = %1" ::"r"(dst), "v"(result)
                 : "memory");
}

static void write_panel(unsigned int nth, unsigned int ith, void* arg) {
    const struct w8a8_panel* p = arg;
    const size_t tile_bytes    = 4 * SLICE_BYTES;
    const uint32_t rows        = (p->rows + 3) / 4;
    const uint32_t cols        = p->nt * 2;
    const uint32_t row_begin   = rows * ith / nth;
    const uint32_t row_end     = rows * (ith + 1) / nth;
    for (uint32_t r = row_begin; r < row_end; ++r) {
        const uint32_t remaining = p->rows - 4 * r;
        const uint8_t* tile      = p->digits + (r / 16) * p->nt * tile_bytes + (r % 16) * 128;
        HVX_Vector row_scale[4];
        for (unsigned i = 0; i < 4; ++i) {
            row_scale[i] = hvx_vec_splat_f32(p->row_scales[r * 4 + i]);
        }
        for (uint32_t c = 0; c < cols; ++c) {
            const uint8_t* v            = tile + (c / 2) * tile_bytes + (c % 2) * SLICE_BYTES / 2;
            const HVX_Vector lo         = *(const HVX_Vector*)v;
            const HVX_Vector mid        = *(const HVX_Vector*)(v + SLICE_BYTES);
            const HVX_Vector hi         = *(const HVX_Vector*)(v + 2 * SLICE_BYTES);
            const HVX_Vector top        = *(const HVX_Vector*)(v + 3 * SLICE_BYTES);
            const HVX_VectorPair x      = Q6_W_vshuff_VVR(mid, lo, -1);
            const HVX_VectorPair y      = Q6_W_vshuff_VVR(top, hi, -1);
            const HVX_VectorPair uv     = Q6_W_vshuff_VVR(Q6_V_lo_W(y), Q6_V_lo_W(x), -2);
            const HVX_VectorPair zw     = Q6_W_vshuff_VVR(Q6_V_hi_W(y), Q6_V_hi_W(x), -2);
            const HVX_Vector scale      = *(const HVX_Vector*)(p->scales + c * 32);
            const HVX_Vector correction = *(const HVX_Vector*)(p->correction + c * 32);
            float* dst                  = p->dst + (size_t)r * 4 * p->ld + c * 32;
            store_scaled(dst, Q6_V_lo_W(uv), scale, row_scale[0], correction);
            if (remaining > 1) {
                store_scaled(dst + p->ld, Q6_V_hi_W(uv), scale, row_scale[1], correction);
            }
            if (remaining > 2) {
                store_scaled(dst + 2 * p->ld, Q6_V_lo_W(zw), scale, row_scale[2], correction);
            }
            if (remaining > 3) {
                store_scaled(dst + 3 * p->ld, Q6_V_hi_W(zw), scale, row_scale[3], correction);
            }
        }
    }
}

int op_matmul_convrot_w8a8(struct htp_ops_context* octx) {
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)
        return HTP_STATUS_OK;

    struct htp_context* ctx       = octx->ctx;
    const struct htp_tensor* w    = octx->src[0];
    const struct htp_tensor* x    = octx->src[1];
    const struct htp_tensor* tail = octx->src[3];
    if (!ctx->hmx_queue || !(w->flags & HTP_TENSOR_REPACK)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    float* dst                   = htp_tensor_data(octx->dst);
    const float* activation      = htp_tensor_data(x);
    const float* activation_tail = tail ? htp_tensor_data(tail) : NULL;
    const uint8_t* b             = htp_tensor_data(w);
    const float* scale           = htp_tensor_data(octx->src[2]);
    const uint32_t k = w->ne[0], n = w->ne[1], k0 = x->ne[0];
    const int32_t* correction = (const int32_t*)(b + (size_t)k * n);
    const uint32_t m          = x->ne[1] * x->ne[2] * x->ne[3];
    const uint32_t group_size = octx->op_params[0];

    work_queue_func_t prepare     = group_size == 256
                                        ? prepare_w4a8_activations_h256
                                        : prepare_w4a8_activations;
    const size_t tile_bytes       = 4 * SLICE_BYTES;
    const size_t a_stride         = 64 * k;
    const size_t a_bytes_per_tile = a_stride + 64 * sizeof(float);
    const size_t b_stride         = 64 * k;
    const size_t matrix_scales    = sizeof(float) * n;
    const size_t fixed            = hex_align_up(4 * BIAS_BYTES + matrix_scales + sizeof(int32_t) * n, 2048);
    const uint32_t mt             = (m + 63) / 64;
    const uint32_t nt             = n / 64;
    uint32_t nb                   = 3;
    if (k <= 4096 && n >= 12288) {
        nb = 4;
    } else if (k <= 4096 && n >= 8192) {
        nb = 8;
    } else if (k >= 12288 && k < 16384) {
        nb = 1;
    } else if (k >= 6144 && k < 12288) {
        nb = 2;
    }
    const uint32_t write_workers = n >= 12288 ? hex_smin(ctx->n_threads, 4) : ctx->n_threads;
    if (nb > nt)
        nb = nt;
    const bool batch_weights    = k <= 4096 && (n <= 8192 || n >= 12288);
    const uint32_t weight_batch = batch_weights ? nb : (k >= 8192 ? 1 : 2);
    uint32_t wb                 = hex_smin(nb, weight_batch);
    while (nb > 1 && fixed + a_bytes_per_tile + 2 * nb * tile_bytes + 2 * wb * b_stride > ctx->vtcm_size) {
        nb /= 2;
        wb = hex_smin(nb, weight_batch);
    }
    const size_t overhead       = fixed + 2 * wb * b_stride;
    const size_t row_tile_bytes = a_bytes_per_tile + 2 * nb * tile_bytes;
    if (overhead + row_tile_bytes > ctx->vtcm_size)
        return HTP_STATUS_VTCM_TOO_SMALL;
    const uint32_t mb = hex_smin(mt, (ctx->vtcm_size - overhead) / row_tile_bytes);

    uint32_t* bias = (uint32_t*)ctx->vtcm_base;
    // The combined CVT is scaled by 1/16. Start at exponent 24 to read all four bytes.
    const uint32_t shifts[4] = {24, 16, 8, 0};
    for (uint32_t s = 0; s < 4; ++s) {
        for (uint32_t j = 0; j < 32; ++j) {
            bias[s * 64 + j]      = shifts[s] << 10;
            bias[s * 64 + 32 + j] = 0;
        }
    }
    float* scales           = (float*)(ctx->vtcm_base + 4 * BIAS_BYTES);
    int32_t* col_correction = (int32_t*)(scales + n);
    memcpy(scales, scale, matrix_scales);
    memcpy(col_correction, correction, sizeof(int32_t) * n);

    uint8_t* a_buf      = ctx->vtcm_base + fixed;
    uint8_t* b_buf[2]   = {a_buf + mb * a_stride, a_buf + mb * a_stride + wb * b_stride};
    uint8_t* out_buf[2] = {b_buf[1] + wb * b_stride,
                           b_buf[1] + wb * b_stride + mb * nb * tile_bytes};
    float* row_scales   = (float*)(out_buf[1] + mb * nb * tile_bytes);
    // The second weight buffer and both CVT buffers are idle during activation preparation.
    const size_t available         = (uint8_t*)row_scales - b_buf[1];
    const uint32_t scratch_workers = available / (4 * sizeof(float) * k);
    const uint32_t act_threads     = scratch_workers < ctx->n_threads ? scratch_workers : ctx->n_threads;
    // Wide output panels with a partial activation tile favor the row pipeline.
    if (group_size == 256 && (nb <= 4 || m % 64 == 0) &&
        available >= (size_t)act_threads * 8 * sizeof(float) * k) {
        prepare = prepare_w4a8_activations_h256_prefetch;
    }
    dma_queue_t dma = ctx->dma[0];

    // Use the unfilled end of the activation panel as scratch for long rows.
    const size_t prefetch_scratch_bytes = ctx->n_threads * 8 * sizeof(float) * k;
    const size_t prepare_space          = (uint8_t*)row_scales - a_buf;
    const bool split_prepare            = group_size == 256 && k >= 8192 &&
                               available < prefetch_scratch_bytes && prepare_space >= prefetch_scratch_bytes + a_stride;
    if (!split_prepare) {
        dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[0], b), wb * b_stride);
    }
    for (uint32_t i = 0; i < mt; i += mb) {
        const uint32_t im             = hex_smin(mb, mt - i);
        const uint32_t rows           = hex_smin(m - i * TILE_ROWS, im * TILE_ROWS);
        const size_t first_row        = (size_t)i * TILE_ROWS;
        struct w4a8_activations panel = {
            .src        = activation + first_row * k0,
            .tail       = activation_tail ? activation_tail + first_row * (k - k0) : NULL,
            .dst        = a_buf,
            .scales     = row_scales,
            .scratch    = (HVX_Vector*)b_buf[1],
            .rows       = rows,
            .k          = k,
            .k0         = k0,
            .dma        = ctx->dma,
            .group_size = group_size,
        };
        if (split_prepare) {
            const uint32_t head_tiles    = hex_smin(im, (prepare_space - prefetch_scratch_bytes) / a_stride);
            struct w4a8_activations head = panel;
            head.rows                    = hex_smin(rows, head_tiles * TILE_ROWS);
            head.scratch                 = (HVX_Vector*)((uint8_t*)row_scales - prefetch_scratch_bytes);
            work_queue_run(ctx->work_queue, prepare_w4a8_activations_h256_prefetch, &head, ctx->n_threads);
            panel.src += (size_t)head.rows * k0;
            if (panel.tail) {
                panel.tail += (size_t)head.rows * (k - k0);
            }
            panel.dst += head_tiles * a_stride;
            panel.scales += head.rows;
            panel.rows -= head.rows;
            dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[0], b), wb * b_stride);
        }
        if (panel.rows) {
            work_queue_run(ctx->work_queue, prepare, &panel, act_threads);
        }
        dma_queue_flush(dma);

        uint32_t weight_slot       = 0;
        uint32_t output_slot       = 0;
        struct w8a8_panel previous = {0};
        for (uint32_t j = 0; j < nt; j += nb, output_slot ^= 1) {
            const uint32_t width     = hex_smin(nb, nt - j);
            const uint32_t chunks    = (width + wb - 1) / wb;
            struct w8a8_panel output = {
                .a          = a_buf,
                .digits     = out_buf[output_slot],
                .bias       = bias,
                .dst        = dst + (size_t)i * TILE_ROWS * n + j * TILE_COLS,
                .scales     = scales + j * TILE_COLS,
                .row_scales = row_scales,
                .correction = col_correction + j * TILE_COLS,
                .mt         = im,
                .rows       = rows,
                .nt         = width,
                .output_nt  = width,
                .k          = k,
                .ld         = n,
            };
            for (uint32_t wj = 0, part = 0; wj < width; wj += wb, ++part, weight_slot ^= 1) {
                struct w8a8_panel job = output;
                job.nt                = hex_smin(wb, width - wj);
                job.b                 = b_buf[weight_slot];
                job.digits += wj * tile_bytes;
                hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(multiply_panel, &job));
                const uint32_t next = j + wj + job.nt;
                if (next < nt) {
                    const uint32_t end = hex_smin(nt, (next / nb + 1) * nb);
                    dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[weight_slot ^ 1], b + next * b_stride),
                                             hex_smin(wb, end - next) * b_stride);
                }
                if (j) {
                    const uint32_t first = im * part / chunks;
                    const uint32_t end   = im * (part + 1) / chunks;
                    if (end > first) {
                        struct w8a8_panel slice = previous;
                        slice.digits += first * previous.nt * tile_bytes;
                        slice.dst += (size_t)first * TILE_ROWS * n;
                        slice.row_scales += first * TILE_ROWS;
                        slice.rows = hex_smin(rows - first * TILE_ROWS, (end - first) * TILE_ROWS);
                        work_queue_run(ctx->work_queue, write_panel, &slice, write_workers);
                    }
                }
                hmx_queue_pop(ctx->hmx_queue);
                dma_queue_flush(dma);
            }
            previous = output;
        }
        if (!split_prepare && i + mb < mt) {
            dma_queue_push_single_1d(dma, dma_make_ptr(b_buf[0], b), wb * b_stride);
        }
        work_queue_run(ctx->work_queue, write_panel, &previous, write_workers);
    }
    return HTP_STATUS_OK;
}
