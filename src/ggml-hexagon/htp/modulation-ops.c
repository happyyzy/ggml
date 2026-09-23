#include <HAP_farf.h>
#include <hexagon_protos.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hex-dma.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-tensor.h"
#include "hvx-utils.h"

struct modulation_context {
    struct htp_ops_context * octx;
    uint32_t features;
    uint32_t tokens;
    uint32_t rows;
    uint32_t split;
    uint32_t block_rows;
    uint32_t row_bytes;
    uint32_t spad_size_per_thread;
    bool has_second;
    const float * params[4];
    uint8_t * src0_spad;
    uint8_t * src1_spad;
    uint8_t * dst_spad;
};

static inline uint64_t modulation_nelements(const struct htp_tensor * tensor) {
    return (uint64_t) tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

static inline void modulation_dma_push(
        dma_queue * queue,
        void *      dst,
        const void * src,
        uint32_t    bytes,
        uint32_t    rows) {
    while (!dma_queue_push(queue, dma_make_ptr(dst, src), bytes, bytes, bytes, rows)) {
        dma_queue_flush(queue);
    }
}

static inline void modulate_row(
        float *       dst,
        const float * src,
        const float * scale,
        const float * shift,
        uint32_t      features) {
    HVX_Vector * out = (HVX_Vector *) dst;
    const HVX_Vector * in = (const HVX_Vector *) src;
    const HVX_Vector * s = (const HVX_Vector *) scale;
    const HVX_Vector * b = (const HVX_Vector *) shift;
    const uint32_t nvec = features / 32u;

    for (uint32_t i = 0; i < nvec; ++i) {
        const HVX_Vector x = in[i];
        out[i] = hvx_vec_add_f32_f32(
            hvx_vec_add_f32_f32(x, hvx_vec_mul_f32_f32(x, s[i])), b[i]);
    }
}

static inline void gated_residual_row(
        float *       dst,
        const float * base,
        const float * branch,
        const float * gate,
        uint32_t      features) {
    HVX_Vector * out = (HVX_Vector *) dst;
    const HVX_Vector * x = (const HVX_Vector *) base;
    const HVX_Vector * update = (const HVX_Vector *) branch;
    const HVX_Vector * g = (const HVX_Vector *) gate;
    const uint32_t nvec = features / 32u;

    for (uint32_t i = 0; i < nvec; ++i) {
        out[i] = hvx_vec_add_f32_f32(x[i], hvx_vec_mul_f32_f32(update[i], g[i]));
    }
}

static void modulation_worker(unsigned int nth, unsigned int ith, void * data) {
    struct modulation_context * mctx = (struct modulation_context *) data;
    struct htp_ops_context * octx = mctx->octx;
    const bool gated = octx->op == HTP_OP_GATED_RESIDUAL;
    const uint32_t first = (uint32_t) ((uint64_t) mctx->rows * ith / nth);
    const uint32_t last = (uint32_t) ((uint64_t) mctx->rows * (ith + 1u) / nth);
    if (first >= last) {
        return;
    }

    dma_queue * queue = octx->ctx->dma[ith];
    const uint32_t half = mctx->spad_size_per_thread / 2u;
    uint8_t * src0_base = mctx->src0_spad + (size_t) ith * mctx->spad_size_per_thread;
    uint8_t * src1_base = gated ? mctx->src1_spad + (size_t) ith * mctx->spad_size_per_thread : NULL;
    uint8_t * dst_base = mctx->dst_spad + (size_t) ith * mctx->spad_size_per_thread;
    uint32_t prefetch = first;
    uint32_t slot = 0;

    for (uint32_t i = 0; i < 2u && prefetch < last; ++i) {
        const uint32_t count = hex_smin(mctx->block_rows, last - prefetch);
        const uint32_t bytes = count * mctx->row_bytes;
        uint8_t * src0_tile = src0_base + slot * half;
        uint8_t * dst_tile = dst_base + slot * half;
        uint8_t * dst_ddr = (uint8_t *) octx->dst->data + (size_t) prefetch * mctx->row_bytes;
        const uint8_t * src0_ddr = (const uint8_t *) octx->src[0]->data + (size_t) prefetch * mctx->row_bytes;

        modulation_dma_push(queue, dst_ddr, dst_tile, bytes, 0);
        modulation_dma_push(queue, src0_tile, src0_ddr, bytes, 1);
        if (gated) {
            uint8_t * src1_tile = src1_base + slot * half;
            const uint8_t * src1_ddr = (const uint8_t *) octx->src[1]->data + (size_t) prefetch * mctx->row_bytes;
            modulation_dma_push(queue, src1_tile, src1_ddr, bytes, 1);
        }
        prefetch += count;
        slot ^= 1u;
    }

    struct htp_thread_trace * trace = &octx->ctx->trace[ith];
    for (uint32_t row = first; row < last;) {
        const uint32_t count = hex_smin(mctx->block_rows, last - row);
        const uint32_t bytes = count * mctx->row_bytes;
        uint8_t * dst_tile = (uint8_t *) dma_queue_pop(queue).src;
        uint8_t * src0_tile = (uint8_t *) dma_queue_pop(queue).dst;
        uint8_t * src1_tile = gated ? (uint8_t *) dma_queue_pop(queue).dst : NULL;

        htp_trace_event_start(trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);
        for (uint32_t r = 0; r < count; ++r) {
            const uint32_t token = (row + r) % mctx->tokens;
            const uint32_t set = mctx->has_second && token >= mctx->split ? 1u : 0u;
            float * dst_row = (float *) (dst_tile + (size_t) r * mctx->row_bytes);
            const float * src0_row = (const float *) (src0_tile + (size_t) r * mctx->row_bytes);
            if (gated) {
                const float * src1_row = (const float *) (src1_tile + (size_t) r * mctx->row_bytes);
                gated_residual_row(dst_row, src0_row, src1_row, mctx->params[set], mctx->features);
            } else {
                modulate_row(dst_row, src0_row, mctx->params[2u * set],
                             mctx->params[2u * set + 1u], mctx->features);
            }
        }
        htp_trace_event_stop(trace, HTP_TRACE_EVT_HVX_COMP, (uint16_t) row);

        uint8_t * dst_ddr = (uint8_t *) octx->dst->data + (size_t) row * mctx->row_bytes;
        modulation_dma_push(queue, dst_ddr, dst_tile, bytes, 1);

        if (prefetch < last) {
            const uint32_t next_count = hex_smin(mctx->block_rows, last - prefetch);
            const uint32_t next_bytes = next_count * mctx->row_bytes;
            const uint8_t * src0_next = (const uint8_t *) octx->src[0]->data + (size_t) prefetch * mctx->row_bytes;
            modulation_dma_push(queue, src0_tile, src0_next, next_bytes, 1);
            if (gated) {
                const uint8_t * src1_next = (const uint8_t *) octx->src[1]->data + (size_t) prefetch * mctx->row_bytes;
                modulation_dma_push(queue, src1_tile, src1_next, next_bytes, 1);
            }
            prefetch += next_count;
        }
        row += count;
    }
    dma_queue_flush(queue);
}

int op_modulation(struct htp_ops_context * octx) {
    const bool gated = octx->op == HTP_OP_GATED_RESIDUAL;
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * param0 = octx->src[gated ? 2 : 1];
    const struct htp_tensor * param1 = octx->src[gated ? 3 : 2];
    const struct htp_tensor * param2 = gated ? NULL : octx->src[3];
    const struct htp_tensor * param3 = gated ? NULL : octx->src[4];
    const struct htp_tensor * dst = octx->dst;

    if (!src0 || !param0 || (!gated && !param1) || !dst ||
        src0->type != HTP_TYPE_F32 || param0->type != HTP_TYPE_F32 ||
        (!gated && param1->type != HTP_TYPE_F32) || dst->type != HTP_TYPE_F32 ||
        !htp_tensor_is_contiguous(src0, sizeof(float)) ||
        !htp_tensor_is_contiguous(dst, sizeof(float)) ||
        !htp_tensor_is_contiguous(param0, sizeof(float)) ||
        (!gated && !htp_tensor_is_contiguous(param1, sizeof(float))) ||
        src0->ne[0] == 0 || src0->ne[0] % 32u != 0 ||
        modulation_nelements(param0) != src0->ne[0] ||
        (!gated && modulation_nelements(param1) != src0->ne[0])) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (int i = 0; i < HTP_OP_MAX_DIMS; ++i) {
        if (src0->ne[i] != dst->ne[i]) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    bool has_second = false;
    if (gated) {
        if (!src1 || src1->type != HTP_TYPE_F32 ||
            !htp_tensor_is_contiguous(src1, sizeof(float))) {
            return HTP_STATUS_NO_SUPPORT;
        }
        for (int i = 0; i < HTP_OP_MAX_DIMS; ++i) {
            if (src1->ne[i] != src0->ne[i]) {
                return HTP_STATUS_NO_SUPPORT;
            }
        }
        has_second = param1 != NULL;
        if (has_second &&
            (param1->type != HTP_TYPE_F32 ||
             !htp_tensor_is_contiguous(param1, sizeof(float)) ||
             modulation_nelements(param1) != src0->ne[0])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    } else {
        if ((param2 == NULL) != (param3 == NULL)) {
            return HTP_STATUS_NO_SUPPORT;
        }
        has_second = param2 != NULL;
        if (has_second &&
            (param2->type != HTP_TYPE_F32 || param3->type != HTP_TYPE_F32 ||
             !htp_tensor_is_contiguous(param2, sizeof(float)) ||
             !htp_tensor_is_contiguous(param3, sizeof(float)) ||
             modulation_nelements(param2) != src0->ne[0] ||
             modulation_nelements(param3) != src0->ne[0])) {
            return HTP_STATUS_NO_SUPPORT;
        }
    }

    const uint32_t split = (uint32_t) octx->op_params[0];
    if (split > src0->ne[1]) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t row_bytes = src0->ne[0] * sizeof(float);
    const uint32_t n_params = gated ? (has_second ? 2u : 1u) : (has_second ? 4u : 2u);
    const uint32_t spad_count = gated ? 3u : 2u;
    const size_t param_bytes = (size_t) n_params * row_bytes;
    if (param_bytes >= octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    uint32_t block_rows = (uint32_t) ((octx->ctx->vtcm_size - param_bytes) /
                                      ((size_t) octx->n_threads * spad_count * 2u * row_bytes));
    block_rows = hex_smin(block_rows, 16u);
    if (block_rows == 0) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    struct modulation_context mctx = {
        .octx = octx,
        .features = src0->ne[0],
        .tokens = src0->ne[1],
        .rows = src0->ne[1] * src0->ne[2] * src0->ne[3],
        .split = split,
        .block_rows = block_rows,
        .row_bytes = row_bytes,
        .spad_size_per_thread = 2u * block_rows * row_bytes,
        .has_second = has_second,
    };

    uint8_t * cursor = octx->ctx->vtcm_base;
    dma_queue * queue = octx->ctx->dma[0];
    const struct htp_tensor * tensors[4] = { param0, param1, param2, param3 };
    for (uint32_t i = 0; i < n_params; ++i) {
        mctx.params[i] = (const float *) cursor;
        modulation_dma_push(queue, cursor, (const void *) tensors[i]->data, row_bytes, 1);
        cursor += row_bytes;
    }
    dma_queue_flush(queue);
    asm volatile("syncht" ::: "memory");

    const size_t spad_bytes = (size_t) octx->n_threads * mctx.spad_size_per_thread;
    mctx.src0_spad = cursor;
    cursor += spad_bytes;
    if (gated) {
        mctx.src1_spad = cursor;
        cursor += spad_bytes;
    }
    mctx.dst_spad = cursor;
    cursor += spad_bytes;
    if ((size_t) (cursor - octx->ctx->vtcm_base) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    work_queue_run(octx->ctx->worker_pool, modulation_worker, &mctx, octx->n_threads);
    asm volatile("syncht" ::: "memory");
    return HTP_STATUS_OK;
}
