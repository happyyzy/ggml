
#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "conv2d-ops.h"
#include "hex-dma.h"
#include "hex-profile.h"
#include "matmul-ops.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#include "hmx-mm-kernels-tiled.h"
#pragma clang diagnostic pop
#include "hmx-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"


struct conv2d_prepare_state {
    struct htp_context * ctx;
    float * x;
    const float * src;
    __fp16 * a;
    uint32_t iw;
    uint32_t ih;
    uint32_t src_w;
    uint32_t src_h;
    uint32_t halo_w;
    uint32_t halo_h;
    uint32_t sx_begin;
    uint32_t sy_begin;
    uint32_t copy_w;
    uint32_t copy_h;
    uint32_t dx;
    uint32_t dy;
    int32_t sx0;
    int32_t sy0;
    uint32_t kh;
    uint32_t kw;
    uint32_t ic_blocks;
    uint32_t n_groups;
    uint32_t group_channels;
    uint32_t k_tiles;
    uint32_t activation_k_tiles;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t x_tiles;
    uint32_t m_tiles;
    bool clear;
    bool wait_weight_dma;
    bool upscale2;
    bool direct_upscale2;
    bool input_f16;
};

static inline HVX_Vector conv2d_join_f16_halves(HVX_Vector lo, HVX_Vector hi) {
    const HVX_VectorPred lower_half = Q6_Q_vsetq_R(64);
    return Q6_V_vmux_QVV(lower_half, lo, Q6_V_vror_VR(hi, 64));
}

static inline HVX_Vector conv2d_pack_f16_pair(HVX_Vector row0, HVX_Vector row1) {
    return Q6_Vh_vshuff_Vh(conv2d_join_f16_halves(row0, row1));
}

static inline void conv2d_upscale2_row(float * dst, const float * src,
                                       uint32_t logical_x, uint32_t count) {
    while (count > 0) {
        const uint32_t take = count < 32u ? count : 32u;
        const HVX_Vector input = hvx_vmemu(src + logical_x / 2u);
        const HVX_VectorPair dup = Q6_W_vshuff_VVR(input, input, -4);
        HVX_Vector output = Q6_V_lo_W(dup);
        if (logical_x & 1u) {
            output = Q6_V_valign_VVR(Q6_V_hi_W(dup), output, sizeof(float));
        }
        hvx_vec_store_u(dst, take * sizeof(float), output);
        logical_x += take;
        dst += take;
        count -= take;
    }
}

static inline void conv2d_upscale2_row_f16(__fp16 * dst, const __fp16 * src,
                                           uint32_t logical_x, uint32_t count) {
    while (count > 0) {
        const uint32_t take = count < 64u ? count : 64u;
        const HVX_Vector input = hvx_vmemu(src + logical_x / 2u);
        const HVX_VectorPair dup = Q6_W_vshuff_VVR(input, input, -2);
        HVX_Vector output = Q6_V_lo_W(dup);
        if (logical_x & 1u) {
            output = Q6_V_valign_VVR(Q6_V_hi_W(dup), output, sizeof(__fp16));
        }
        hvx_vec_store_u(dst, take * sizeof(__fp16), output);
        logical_x += take;
        dst += take;
        count -= take;
    }
}

static void conv2d_push_halo_group(
        const struct conv2d_prepare_state * st,
        dma_queue * q,
        uint32_t group,
        void * x_ptr) {
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    if (st->input_f16) {
        __fp16 * x = (__fp16 *) x_ptr;
        if (st->clear) {
            memset(x, 0, st->group_channels * channel_elms * sizeof(__fp16));
        }
        const uint32_t channel_first = group * st->group_channels;
        if (st->upscale2) {
            for (uint32_t c = 0; c < st->group_channels; ++c) {
                __fp16 * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
                const __fp16 * channel = (const __fp16 *) st->src +
                    (size_t) (channel_first + c) * st->src_h * st->src_w;
                for (uint32_t y = 0; y < st->copy_h; ++y) {
                    const uint32_t logical_y = st->sy_begin + y;
                    const __fp16 * s = channel + (size_t) (logical_y / 2u) * st->src_w;
                    conv2d_upscale2_row_f16(d + (size_t) y * st->halo_w, s,
                                            st->sx_begin, st->copy_w);
                }
            }
            return;
        }
        for (uint32_t c = 0; c < st->group_channels; ++c) {
            __fp16 * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
            const __fp16 * s = (const __fp16 *) st->src +
                ((size_t) (channel_first + c) * st->ih + st->sy_begin) * st->iw +
                st->sx_begin;
            while (!dma_queue_push(q, dma_make_ptr(d, s),
                                   st->halo_w * sizeof(__fp16), st->iw * sizeof(__fp16),
                                   st->copy_w * sizeof(__fp16), st->copy_h)) {
                dma_queue_flush(q);
            }
        }
        return;
    }
    float * x = (float *) x_ptr;
    if (st->clear) {
        memset(x, 0, st->group_channels * channel_elms * sizeof(float));
    }
    const uint32_t channel_first = group * st->group_channels;
    if (st->upscale2) {
        for (uint32_t c = 0; c < st->group_channels; ++c) {
            float * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
            const float * channel = st->src +
                (size_t) (channel_first + c) * st->src_h * st->src_w;
            for (uint32_t y = 0; y < st->copy_h; ++y) {
                const uint32_t logical_y = st->sy_begin + y;
                const float * s = channel + (size_t) (logical_y / 2u) * st->src_w;
                conv2d_upscale2_row(d + (size_t) y * st->halo_w, s,
                                    st->sx_begin, st->copy_w);
            }
        }
        return;
    }
    for (uint32_t c = 0; c < st->group_channels; ++c) {
        float * d = x + ((size_t) c * st->halo_h + st->dy) * st->halo_w + st->dx;
        const float * s = st->src + ((size_t) (channel_first + c) * st->ih + st->sy_begin) * st->iw + st->sx_begin;
        while (!dma_queue_push(q, dma_make_ptr(d, s),
                               st->halo_w * sizeof(float), st->iw * sizeof(float),
                               st->copy_w * sizeof(float), st->copy_h)) {
            dma_queue_flush(q);
        }
    }
}

static inline void conv2d_store_activation_triplet(
        const struct conv2d_prepare_state * st,
        uint32_t source_mt,
        uint32_t icb,
        uint32_t cp,
        HVX_Vector p0,
        HVX_Vector p1,
    HVX_Vector p2) {
    HVX_Vector * tile0 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         0 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    HVX_Vector * tile1 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         1 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    HVX_Vector * tile2 = (HVX_Vector *)
        (st->a + ((size_t) source_mt * st->activation_k_tiles +
         2 * st->ic_blocks + icb) * HTP_MM_HMX_TILE_N_ELMS);
    tile0[cp] = p0;
    tile1[cp] = p1;
    tile2[cp] = p2;
}

static void conv2d_pack_activation_group_f16(
        const struct conv2d_prepare_state * st,
        uint32_t group,
        __fp16 * x) {
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32u;
    const uint32_t cp_first = (channel_first % 32u) / 2u;
    const uint32_t n_pairs = st->group_channels / 2u;
    if (st->kw == 3) {
        for (uint32_t cp = 0; cp < n_pairs; ++cp) {
            for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
                const __fp16 * row = x + (size_t) sy * st->halo_w;
                const __fp16 * row0 = row + (2u * cp + 0u) * channel_elms;
                const __fp16 * row1 = row + (2u * cp + 1u) * channel_elms;
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t x0 = tx * HTP_CONV2D_TILE_W;
                    conv2d_store_activation_triplet(
                        st, sy * st->x_tiles + tx, icb, cp_first + cp,
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 0u),
                                             hvx_vmemu(row1 + x0 + 0u)),
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 1u),
                                             hvx_vmemu(row1 + x0 + 1u)),
                        conv2d_pack_f16_pair(hvx_vmemu(row0 + x0 + 2u),
                                             hvx_vmemu(row1 + x0 + 2u)));
                }
            }
        }
        return;
    }
    for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
        for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
            const uint32_t mt = oy * st->x_tiles + tx;
            HVX_Vector * tile = (HVX_Vector *)
                (st->a + ((size_t) mt * st->k_tiles + icb) *
                 HTP_MM_HMX_TILE_N_ELMS);
            const __fp16 * src = x + (size_t) oy * st->halo_w +
                                  tx * HTP_CONV2D_TILE_W;
            for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                tile[cp_first + cp] = conv2d_pack_f16_pair(
                    hvx_vmemu(src + (2u * cp + 0u) * channel_elms),
                    hvx_vmemu(src + (2u * cp + 1u) * channel_elms));
            }
        }
    }
}

static inline HVX_Vector conv2d_upscale2_shift(
        HVX_Vector lo,
        HVX_Vector hi,
        uint32_t shift_words) {
    return shift_words == 0 ? lo
                            : Q6_V_valign_VVR(hi, lo, shift_words * sizeof(float));
}

static void conv2d_prepare_upscale_group_direct(
        const struct conv2d_prepare_state * st,
        uint32_t group) {
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32;
    const uint32_t cp_first = (channel_first % 32) / 2;
    const uint32_t n_pairs = st->group_channels / 2;
    const uint32_t x_parity = (uint32_t) st->sx0 & 1u;

    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
        const uint32_t c0 = channel_first + 2 * cp;
        const uint32_t c1 = c0 + 1;
        for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
            const uint32_t iy = (uint32_t) (st->sy0 + (int32_t) sy) / 2u;
            const float * row0 = st->src + ((size_t) c0 * st->src_h + iy) * st->src_w;
            const float * row1 = st->src + ((size_t) c1 * st->src_h + iy) * st->src_w;
            for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                const uint32_t logical_x = (uint32_t) st->sx0 + tx * HTP_CONV2D_TILE_W;
                const uint32_t ix = logical_x / 2u;
                const HVX_Vector in0 = hvx_vmemu(row0 + ix);
                const HVX_Vector in1 = hvx_vmemu(row1 + ix);
                const HVX_VectorPair dup0 = Q6_W_vshuff_VVR(in0, in0, -4);
                const HVX_VectorPair dup1 = Q6_W_vshuff_VVR(in1, in1, -4);
                HVX_Vector x00 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 0u);
                HVX_Vector x01 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 0u);
                HVX_Vector x10 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 1u);
                HVX_Vector x11 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 1u);
                HVX_Vector x20 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 2u);
                HVX_Vector x21 = conv2d_upscale2_shift(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 2u);
                conv2d_store_activation_triplet(
                    st, sy * st->x_tiles + tx, icb, cp_first + cp,
                    hvx_vec_f32_to_f16_shuff(x00, x01),
                    hvx_vec_f32_to_f16_shuff(x10, x11),
                    hvx_vec_f32_to_f16_shuff(x20, x21));
            }
        }
    }
}

static inline HVX_Vector conv2d_upscale2_shift_f16(
        HVX_Vector lo,
        HVX_Vector hi,
        uint32_t shift) {
    return shift == 0 ? lo : Q6_V_valign_VVR(hi, lo, shift * sizeof(__fp16));
}

static void conv2d_prepare_upscale_group_direct_f16(
        const struct conv2d_prepare_state * st,
        uint32_t group) {
    const uint32_t channel_first = group * st->group_channels;
    const uint32_t icb = channel_first / 32u;
    const uint32_t cp_first = (channel_first % 32u) / 2u;
    const uint32_t n_pairs = st->group_channels / 2u;
    const uint32_t x_parity = (uint32_t) st->sx0 & 1u;
    const __fp16 * src = (const __fp16 *) st->src;

    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
        const uint32_t c0 = channel_first + 2u * cp;
        const uint32_t c1 = c0 + 1u;
        for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
            const uint32_t iy = (uint32_t) (st->sy0 + (int32_t) sy) / 2u;
            const __fp16 * row0 = src + ((size_t) c0 * st->src_h + iy) * st->src_w;
            const __fp16 * row1 = src + ((size_t) c1 * st->src_h + iy) * st->src_w;
            for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                const uint32_t logical_x = (uint32_t) st->sx0 + tx * HTP_CONV2D_TILE_W;
                const uint32_t ix = logical_x / 2u;
                const HVX_Vector in0 = hvx_vmemu(row0 + ix);
                const HVX_Vector in1 = hvx_vmemu(row1 + ix);
                const HVX_VectorPair dup0 = Q6_W_vshuff_VVR(in0, in0, -2);
                const HVX_VectorPair dup1 = Q6_W_vshuff_VVR(in1, in1, -2);
                HVX_Vector x00 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 0u);
                HVX_Vector x01 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 0u);
                HVX_Vector x10 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 1u);
                HVX_Vector x11 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 1u);
                HVX_Vector x20 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup0), Q6_V_hi_W(dup0), x_parity + 2u);
                HVX_Vector x21 = conv2d_upscale2_shift_f16(
                    Q6_V_lo_W(dup1), Q6_V_hi_W(dup1), x_parity + 2u);
                conv2d_store_activation_triplet(
                    st, sy * st->x_tiles + tx, icb, cp_first + cp,
                    conv2d_pack_f16_pair(x00, x01),
                    conv2d_pack_f16_pair(x10, x11),
                    conv2d_pack_f16_pair(x20, x21));
            }
        }
    }
}

static void conv2d_prepare_activation_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_prepare_state * st = (struct conv2d_prepare_state *) data;
    const uint32_t group_first = fastdiv(st->n_groups * ith, &st->ctx->n_threads_div);
    const uint32_t group_last = fastdiv(st->n_groups * (ith + 1), &st->ctx->n_threads_div);
    const size_t channel_elms = (size_t) st->halo_w * st->halo_h;
    const uint32_t element_size = st->input_f16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t x_slot_first = 2 * ith;
    uint8_t * x_slots = (uint8_t *) st->x +
        (size_t) x_slot_first * st->group_channels * channel_elms * element_size;
    dma_queue * q = st->ctx->dma[ith];
    if (ith == 0 && st->wait_weight_dma) {
        dma_queue_pop(q);
    }
    if (group_first == group_last) {
        return;
    }
    if (st->direct_upscale2) {
        for (uint32_t group = group_first; group < group_last; ++group) {
            if (st->input_f16) {
                conv2d_prepare_upscale_group_direct_f16(st, group);
            } else {
                conv2d_prepare_upscale_group_direct(st, group);
            }
        }
        return;
    }

    if (st->copy_w != 0 && st->copy_h != 0) {
        conv2d_push_halo_group(st, q, group_first, x_slots);
        if (group_first + 1 < group_last) {
            conv2d_push_halo_group(st, q, group_first + 1,
                                   x_slots + st->group_channels * channel_elms * element_size);
        }
    } else if (st->clear) {
        memset(x_slots, 0, (group_last - group_first < 2 ? group_last - group_first : 2) *
                            st->group_channels * channel_elms * element_size);
    }

    for (uint32_t group = group_first; group < group_last; ++group) {
        if (!st->upscale2 && st->copy_w != 0 && st->copy_h != 0) {
            for (uint32_t c = 0; c < st->group_channels; ++c) {
                dma_queue_pop(q);
            }
        }

        uint8_t * x_ptr = x_slots + (size_t) ((group - group_first) & 1u) *
                                   st->group_channels * channel_elms * element_size;
        if (st->input_f16) {
            conv2d_pack_activation_group_f16(st, group, (__fp16 *) x_ptr);
        } else {
            const uint32_t channel_first = group * st->group_channels;
            const uint32_t icb = channel_first / 32;
            const uint32_t cp_first = (channel_first % 32) / 2;
            const uint32_t n_pairs = st->group_channels / 2;
            float * x = (float *) x_ptr;
            if (st->kw == 3) {
            for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                for (uint32_t sy = 0; sy < st->halo_h; ++sy) {
                    const float * row = x + (size_t) sy * st->halo_w;
                        const float * row0 = row + (2 * cp + 0) * channel_elms;
                        const float * row1 = row + (2 * cp + 1) * channel_elms;
                        HVX_Vector lo0 = hvx_vmemu(row0);
                        HVX_Vector lo1 = hvx_vmemu(row1);
                        for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                            HVX_Vector hi0 = hvx_vmemu(
                                row0 + (tx + 1) * HTP_CONV2D_TILE_W);
                            HVX_Vector hi1 = hvx_vmemu(
                                row1 + (tx + 1) * HTP_CONV2D_TILE_W);
                            const HVX_Vector p0 = hvx_vec_f32_to_f16_shuff(lo0, lo1);
                            const HVX_Vector p1 = hvx_vec_f32_to_f16_shuff(
                                Q6_V_valign_VVR(hi0, lo0, sizeof(float)),
                                Q6_V_valign_VVR(hi1, lo1, sizeof(float)));
                            const HVX_Vector p2 = hvx_vec_f32_to_f16_shuff(
                                Q6_V_valign_VVR(hi0, lo0, 2 * sizeof(float)),
                                Q6_V_valign_VVR(hi1, lo1, 2 * sizeof(float)));
                            conv2d_store_activation_triplet(
                                st, sy * st->x_tiles + tx, icb,
                                cp_first + cp, p0, p1, p2);
                            lo0 = hi0;
                            lo1 = hi1;
                        }
                    }
            }
            } else {
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t mt = oy * st->x_tiles + tx;
                    HVX_Vector * tile = (HVX_Vector *)
                        (st->a + ((size_t) mt * st->k_tiles + icb) *
                         HTP_MM_HMX_TILE_N_ELMS);
                    const float * src = x + (size_t) oy * st->halo_w +
                                        tx * HTP_CONV2D_TILE_W;
                    for (uint32_t cp = 0; cp < n_pairs; ++cp) {
                        HVX_Vector row0 =
                            hvx_vmemu(src + (2 * cp + 0) * channel_elms);
                        HVX_Vector row1 =
                            hvx_vmemu(src + (2 * cp + 1) * channel_elms);
                        tile[cp_first + cp] =
                            hvx_vec_f32_to_f16_shuff(row0, row1);
                    }
                }
            }
            }
        }
        if (st->copy_w != 0 && st->copy_h != 0 && group + 2 < group_last) {
            conv2d_push_halo_group(st, q, group + 2, x_ptr);
        }
    }
}

struct conv2d_tile {
    uint32_t x0;
    uint32_t y0;
    uint32_t valid_w;
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t x_tiles;
    uint32_t m_tiles;
};

static inline struct conv2d_tile conv2d_get_tile(
        const struct htp_conv2d_kernel_params * kp,
        uint32_t ow,
        uint32_t oh,
        uint32_t nx,
        uint32_t block) {
    const uint32_t x0 = (block % nx) * kp->tile_w;
    const uint32_t y0 = (block / nx) * kp->tile_h;
    const uint32_t valid_w = hex_smin(kp->tile_w, ow - x0);
    const uint32_t tile_w = htp_conv2d_align_up(valid_w, HTP_CONV2D_TILE_W);
    const uint32_t tile_h = hex_smin(kp->tile_h, oh - y0);
    const uint32_t x_tiles = tile_w / HTP_CONV2D_TILE_W;
    return (struct conv2d_tile) {
        .x0 = x0,
        .y0 = y0,
        .valid_w = valid_w,
        .tile_w = tile_w,
        .tile_h = tile_h,
        .x_tiles = x_tiles,
        .m_tiles = x_tiles * tile_h,
    };
}

struct conv2d_store_state {
    struct htp_context * ctx;
    const __fp16 * c;
    float * dst;
    const float * bias;
    const float * residual;
    uint32_t ow;
    uint32_t oh;
    uint32_t oc;
    uint32_t x0;
    uint32_t y0;
    uint32_t valid_w;
    uint32_t tile_h;
    uint32_t x_tiles;
    uint32_t m_tiles;
    uint32_t channel_offset;
    uint32_t residual_w;
    uint32_t residual_h;
    uint32_t residual_channel_ratio;
    uint32_t residual_factor_t;
    bool final_block;
    atomic_uint * next_pair;
    uint32_t flags;
    bool output_f16;
};

struct conv2d_store_context {
    struct htp_context * ctx;
    float * dst;
    const float * bias;
    const float * residual;
    size_t output_plane;
    uint32_t ow;
    uint32_t oh;
    uint32_t residual_w;
    uint32_t residual_h;
    uint32_t residual_channel_ratio;
    uint32_t residual_factor_t;
    uint32_t flags;
    bool output_f16;
};

static inline struct conv2d_store_state conv2d_make_store_state(
        const struct conv2d_store_context * ctx,
        const struct conv2d_tile * tile,
        const __fp16 * c,
        uint32_t oc_first,
        uint32_t oc_count,
        bool final_block) {
    float * dst = ctx->output_f16
        ? (float *) ((__fp16 *) ctx->dst + (size_t) oc_first * ctx->output_plane)
        : ctx->dst + (size_t) oc_first * ctx->output_plane;
    const float * residual = ctx->residual
        ? ((ctx->flags & HTP_CONV2D_DUP_UP_RESIDUAL)
               ? ctx->residual
               : ctx->residual + (size_t) oc_first * ctx->output_plane)
        : NULL;
    return (struct conv2d_store_state) {
        .ctx = ctx->ctx,
        .c = c,
        .dst = dst,
        .bias = ctx->bias ? ctx->bias + oc_first : NULL,
        .residual = residual,
        .ow = ctx->ow,
        .oh = ctx->oh,
        .oc = oc_count,
        .x0 = tile->x0,
        .y0 = tile->y0,
        .valid_w = tile->valid_w,
        .tile_h = tile->tile_h,
        .x_tiles = tile->x_tiles,
        .m_tiles = tile->m_tiles,
        .channel_offset = oc_first,
        .residual_w = ctx->residual_w,
        .residual_h = ctx->residual_h,
        .residual_channel_ratio = ctx->residual_channel_ratio,
        .residual_factor_t = ctx->residual_factor_t,
        .final_block = final_block,
        .next_pair = NULL,
        .flags = ctx->flags,
        .output_f16 = ctx->output_f16,
    };
}

static inline HVX_Vector conv2d_dup_up_residual(
        const struct conv2d_store_state * st,
        uint32_t channel,
        uint32_t y,
        uint32_t x) {
    const uint32_t global_channel = st->channel_offset + channel;
    const uint32_t source_channel = st->residual_factor_t == 2
        ? global_channel * st->residual_channel_ratio + st->residual_channel_ratio - 1u
        : global_channel * 2u + (y & 1u);
    const float * src = st->residual +
        ((size_t) source_channel * st->residual_h + y / 2u) * st->residual_w + x / 2u;
    const HVX_Vector input = hvx_vmemu(src);
    const HVX_VectorPair duplicated = Q6_W_vshuff_VVR(input, input, -4);
    return Q6_V_lo_W(duplicated);
}

static void conv2d_store_output_f16_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_store_state * st = (struct conv2d_store_state *) data;
    const uint32_t n_pairs = (st->oc + 1) / 2;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPred all = Q6_Q_vcmp_eq_VbVb(zero, zero);
    const uint32_t pair_chunk = 1;
    for (;;) {
        const uint32_t pair_first = atomic_fetch_add_explicit(
            st->next_pair, pair_chunk, memory_order_relaxed);
        if (pair_first >= n_pairs) {
            break;
        }
        const uint32_t pair_last = hex_smin(pair_first + pair_chunk, n_pairs);
        for (uint32_t pair = pair_first; pair < pair_last; ++pair) {
            const uint32_t channel = pair * 2u;
            const uint32_t nt = channel / 32u;
            const uint32_t rp = (channel % 32u) / 2u;
            const HVX_Vector bias0 = hvx_vec_splat_f16(
                (st->flags & HTP_CONV2D_BIAS) ? st->bias[channel] : 0.0f);
            const HVX_Vector bias1 = hvx_vec_splat_f16(
                (st->flags & HTP_CONV2D_BIAS) && channel + 1u < st->oc
                    ? st->bias[channel + 1u] : 0.0f);
            const HVX_Vector packed_bias = Q6_Vh_vshuff_Vh(
                conv2d_join_f16_halves(bias0, bias1));
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                for (uint32_t tx = 0; tx < st->x_tiles; tx += 2u) {
                    HVX_Vector planar[2];
                    const uint32_t count = hex_smin(st->x_tiles - tx, 2u);
                    const uint32_t valid_count = hex_smin(
                        st->valid_w - tx * HTP_CONV2D_TILE_W,
                        count * HTP_CONV2D_TILE_W);
                    for (uint32_t j = 0; j < count; ++j) {
                        const uint32_t mt = oy * st->x_tiles + tx + j;
                        const HVX_Vector * tile = (const HVX_Vector *)
                            (st->c + ((size_t) nt * st->m_tiles + mt) *
                             HTP_MM_HMX_TILE_N_ELMS);
                        HVX_Vector output = hvx_vec_add_f16_f16(tile[rp], packed_bias);
                        planar[j] = Q6_Vh_vdeal_Vh(output);
                    }
                    __fp16 * dst0 = (__fp16 *) st->dst +
                        ((size_t) channel * st->oh + st->y0 + oy) * st->ow +
                        st->x0 + tx * HTP_CONV2D_TILE_W;
                    const HVX_Vector row0 = count == 2u
                        ? conv2d_join_f16_halves(planar[0], planar[1]) : planar[0];
                    if (valid_count == 2u * HTP_CONV2D_TILE_W &&
                        ((uintptr_t) dst0 & 127u) == 0) {
                        Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst0, row0);
                    } else {
                        hvx_vec_store_u(dst0, valid_count * sizeof(__fp16), row0);
                    }
                    if (channel + 1u < st->oc) {
                        __fp16 * dst1 = dst0 + (size_t) st->oh * st->ow;
                        const HVX_Vector p0 = Q6_V_vror_VR(planar[0], 64);
                        const HVX_Vector p1 = count == 2u
                            ? Q6_V_vror_VR(planar[1], 64) : p0;
                        const HVX_Vector row1 = count == 2u
                            ? conv2d_join_f16_halves(p0, p1) : p0;
                        if (valid_count == 2u * HTP_CONV2D_TILE_W &&
                            ((uintptr_t) dst1 & 127u) == 0) {
                            Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst1, row1);
                        } else {
                            hvx_vec_store_u(dst1, valid_count * sizeof(__fp16), row1);
                        }
                    }
                }
            }
        }
    }
    if (st->final_block) {
        asm volatile("syncht" ::: "memory");
    }
}

static void conv2d_store_output_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_store_state * st = (struct conv2d_store_state *) data;
    if (st->output_f16) {
        conv2d_store_output_f16_worker(nth, ith, data);
        return;
    }
    const uint32_t n_pairs = (st->oc + 1) / 2;
    const HVX_Vector zero = Q6_V_vzero();
    const HVX_VectorPred all = Q6_Q_vcmp_eq_VbVb(zero, zero);

    const uint32_t pair_chunk = 4;
    for (;;) {
        const uint32_t pair_first = atomic_fetch_add_explicit(
            st->next_pair, pair_chunk, memory_order_relaxed);
        if (pair_first >= n_pairs) {
            break;
        }
        const uint32_t pair_last = hex_smin(pair_first + pair_chunk, n_pairs);
        for (uint32_t pair = pair_first; pair < pair_last; ++pair) {
            const uint32_t channel = pair * 2;
            const uint32_t nt = channel / 32;
            const uint32_t rp = (channel % 32) / 2;
            const HVX_Vector bias0 = hvx_vec_splat_f32(
                (st->flags & HTP_CONV2D_BIAS) ? st->bias[channel] : 0.0f);
            const HVX_Vector bias1 = hvx_vec_splat_f32(
                (st->flags & HTP_CONV2D_BIAS) && channel + 1 < st->oc
                    ? st->bias[channel + 1] : 0.0f);
            const size_t plane = (size_t) st->oh * st->ow;
            const bool direct_residual = (st->flags & HTP_CONV2D_RESIDUAL) != 0;
            const bool dup_up_residual = (st->flags & HTP_CONV2D_DUP_UP_RESIDUAL) != 0;
            const float * residual0 = direct_residual
                ? st->residual + (size_t) channel * plane +
                    (size_t) st->y0 * st->ow + st->x0
                : NULL;
            const float * residual1 = residual0 && channel + 1 < st->oc
                ? residual0 + plane : NULL;
            if (residual0) {
                const uint32_t rows = hex_smin(st->tile_h, 4u);
                hex_l2fetch(residual0, st->valid_w * sizeof(float),
                            st->ow * sizeof(float), rows);
                if (residual1) {
                    hex_l2fetch(residual1, st->valid_w * sizeof(float),
                                st->ow * sizeof(float), rows);
                }
            }
            for (uint32_t oy = 0; oy < st->tile_h; ++oy) {
                if (residual0 && (oy & 3u) == 0 && oy + 4u < st->tile_h) {
                    const uint32_t rows = hex_smin(st->tile_h - oy - 4u, 4u);
                    hex_l2fetch(residual0 + (size_t) (oy + 4u) * st->ow,
                                st->valid_w * sizeof(float), st->ow * sizeof(float), rows);
                    if (residual1) {
                        hex_l2fetch(residual1 + (size_t) (oy + 4u) * st->ow,
                                    st->valid_w * sizeof(float), st->ow * sizeof(float), rows);
                    }
                }
                for (uint32_t tx = 0; tx < st->x_tiles; ++tx) {
                    const uint32_t valid_count = hex_smin(
                        st->valid_w - tx * HTP_CONV2D_TILE_W,
                        HTP_CONV2D_TILE_W);
                    const uint32_t mt = oy * st->x_tiles + tx;
                    const HVX_Vector * tile = (const HVX_Vector *)
                        (st->c + ((size_t) nt * st->m_tiles + mt) *
                         HTP_MM_HMX_TILE_N_ELMS);
                    const HVX_VectorPair rows = hvx_vec_f16_to_f32_shuff(tile[rp]);
                    float * dst = st->dst + ((size_t) channel * st->oh + st->y0 + oy) *
                                  st->ow + st->x0 + tx * HTP_CONV2D_TILE_W;
                    HVX_Vector row0 = hvx_vec_add_f32_f32(Q6_V_lo_W(rows), bias0);
                    if (dup_up_residual) {
                        row0 = hvx_vec_add_f32_f32(
                            row0, conv2d_dup_up_residual(
                                st, channel, st->y0 + oy,
                                st->x0 + tx * HTP_CONV2D_TILE_W));
                    } else if (direct_residual) {
                        row0 = hvx_vec_add_f32_f32(
                            row0, hvx_vmemu(st->residual + (dst - st->dst)));
                    }
                    if (valid_count == HTP_CONV2D_TILE_W &&
                        ((uintptr_t) dst & 127u) == 0) {
                        Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst, row0);
                    } else {
                        hvx_vec_store_u(dst, valid_count * sizeof(float), row0);
                    }
                    if (channel + 1 < st->oc) {
                        float * dst1 = dst + (size_t) st->oh * st->ow;
                        HVX_Vector row1 = hvx_vec_add_f32_f32(Q6_V_hi_W(rows), bias1);
                        if (dup_up_residual) {
                            row1 = hvx_vec_add_f32_f32(
                                row1, conv2d_dup_up_residual(
                                    st, channel + 1u, st->y0 + oy,
                                    st->x0 + tx * HTP_CONV2D_TILE_W));
                        } else if (direct_residual) {
                            row1 = hvx_vec_add_f32_f32(
                                row1, hvx_vmemu(st->residual + (dst1 - st->dst)));
                        }
                        if (valid_count == HTP_CONV2D_TILE_W &&
                            ((uintptr_t) dst1 & 127u) == 0) {
                            Q6_vmem_QRIV_nt(all, (HVX_Vector *) dst1, row1);
                        } else {
                            hvx_vec_store_u(dst1, valid_count * sizeof(float), row1);
                        }
                    }
                }
            }
        }
    }
    if (st->final_block) {
        asm volatile("syncht" ::: "memory");
    }
}

struct conv2d_hmx_job {
    __fp16 * c;
    const __fp16 * a;
    const __fp16 * b;
    const __fp16 * scales;
    uint32_t mt;
    uint32_t nt;
    uint32_t kt;
    uint32_t kh;
    uint32_t x_tiles;
    uint32_t tile_h;
    uint32_t ic_blocks;
};

static inline struct conv2d_hmx_job conv2d_make_hmx_job(
        __fp16 * c,
        const __fp16 * weight,
        const __fp16 * activation,
        const __fp16 * scales,
        uint32_t n_tiles,
        const struct conv2d_tile * tile,
        const struct htp_conv2d_kernel_params * kp) {
    return (struct conv2d_hmx_job) {
        .c = c,
        .a = weight,
        .b = activation,
        .scales = scales,
        .mt = n_tiles,
        .nt = tile->m_tiles,
        .kt = kp->k_tiles,
        .kh = kp->kh,
        .x_tiles = tile->x_tiles,
        .tile_h = tile->tile_h,
        .ic_blocks = kp->ic_padded / 32u,
    };
}

static inline void conv2d_hmx_mpy_tiles(
        const __fp16 ** row,
        const __fp16 ** col,
        uint32_t n_tiles) {
    while (n_tiles >= 32) {
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :
                     : "r"(65535), "r"(*row), "r"(*col));
        *row += 32 * HTP_MM_HMX_TILE_N_ELMS;
        *col += 32 * HTP_MM_HMX_TILE_N_ELMS;
        n_tiles -= 32;
    }
    if (n_tiles > 0) {
        const uint32_t range = 2048u * n_tiles - 1u;
        asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") :
                     : "r"(range), "r"(*row), "r"(*col));
    }
}

static void conv2d_hmx_3x3(const struct conv2d_hmx_job * job) {
    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) job->scales));

    const uint32_t activation_k_tiles = 3 * job->ic_blocks;
    const size_t activation_stride =
        (size_t) activation_k_tiles * HTP_MM_HMX_TILE_N_ELMS;
    const size_t weight_row_stride =
        (size_t) 3 * activation_stride;
    const uint32_t m_tiles = job->x_tiles * job->tile_h;

    for (uint32_t r = 0; r < job->mt; ++r) {
        const __fp16 * weight_row = job->a + r * weight_row_stride;
        for (uint32_t oy = 0; oy < job->tile_h; ++oy) {
            for (uint32_t tx = 0; tx < job->x_tiles; ++tx) {
                asm volatile(HMX_CLRACC_F16());
                for (uint32_t ky = 0; ky < 3; ++ky) {
                    const __fp16 * row = weight_row + ky * activation_stride;
                    const __fp16 * col = job->b +
                        ((size_t) (oy + ky) * job->x_tiles + tx) * activation_stride;
                    conv2d_hmx_mpy_tiles(&row, &col, activation_k_tiles);
                }
                __fp16 * out = job->c +
                    ((size_t) r * m_tiles + oy * job->x_tiles + tx) *
                    HTP_MM_HMX_TILE_N_ELMS;
                asm volatile(HMX_STORE_AFTER_F16("%0", "%1") :
                             : "r"(out), "r"(0) : "memory");
            }
        }
    }
}

static void conv2d_hmx_worker(void * data) {
    struct conv2d_hmx_job * job = (struct conv2d_hmx_job *) data;
    if (job->kh == 3) {
        conv2d_hmx_3x3(job);
    } else {
        core_dot_chunk_fp16(job->c, job->a, job->b, job->scales,
                            job->mt, job->nt, job->kt);
    }
}

static void conv2d_init_prepare_state(
        struct htp_ops_context * octx,
        float * x,
        __fp16 * a,
        uint32_t x0,
        uint32_t y0,
        uint32_t tile_w,
        uint32_t tile_h,
        bool wait_weight_dma,
        const struct htp_conv2d_kernel_params * kp,
        struct conv2d_prepare_state * state) {
    const struct htp_tensor * src = octx->src[1];
    const int32_t p0 = octx->op_params[2];
    const int32_t p1 = octx->op_params[3];
    const bool upscale2 = (kp->flags & HTP_CONV2D_UPSCALE2) != 0;
    const uint32_t iw = upscale2 ? octx->dst->ne[0] : src->ne[0];
    const uint32_t ih = upscale2 ? octx->dst->ne[1] : src->ne[1];
    const uint32_t halo_w = tile_w + octx->src[0]->ne[0] - 1;
    const uint32_t halo_h = tile_h + octx->src[0]->ne[1] - 1;
    const int32_t sx0 = (int32_t) x0 - p0;
    const int32_t sy0 = (int32_t) y0 - p1;
    const int32_t sx_begin = sx0 < 0 ? 0 : sx0;
    const int32_t sy_begin = sy0 < 0 ? 0 : sy0;
    const int32_t sx_end = hex_smin((int32_t) iw, sx0 + (int32_t) halo_w);
    const int32_t sy_end = hex_smin((int32_t) ih, sy0 + (int32_t) halo_h);

    *state = (struct conv2d_prepare_state) {
        .ctx = octx->ctx,
        .x = x,
        .src = (const float *) src->data,
        .a = a,
        .iw = iw,
        .ih = ih,
        .src_w = src->ne[0],
        .src_h = src->ne[1],
        .halo_w = halo_w,
        .halo_h = halo_h,
        .sx_begin = sx_begin,
        .sy_begin = sy_begin,
        .copy_w = sx_end > sx_begin ? (uint32_t) (sx_end - sx_begin) : 0,
        .copy_h = sy_end > sy_begin ? (uint32_t) (sy_end - sy_begin) : 0,
        .dx = (uint32_t) (sx_begin - sx0),
        .dy = (uint32_t) (sy_begin - sy0),
        .sx0 = sx0,
        .sy0 = sy0,
        .kh = kp->kh,
        .kw = kp->kw,
        .ic_blocks = kp->ic_padded / 32,
        .n_groups = kp->ic / kp->activation_group_channels,
        .group_channels = kp->activation_group_channels,
        .k_tiles = kp->k_tiles,
        .activation_k_tiles = kp->activation_k_tiles,
        .tile_w = tile_w,
        .tile_h = tile_h,
        .x_tiles = tile_w / HTP_CONV2D_TILE_W,
        .m_tiles = (tile_w / HTP_CONV2D_TILE_W) * tile_h,
        .clear = sx_end - sx_begin != (int32_t) halo_w ||
                 sy_end - sy_begin != (int32_t) halo_h,
        .wait_weight_dma = wait_weight_dma,
        .upscale2 = upscale2,
        .direct_upscale2 = upscale2 && sx0 >= 0 && sy0 >= 0 &&
                           sx0 + (int32_t) halo_w <= (int32_t) iw &&
                           sy0 + (int32_t) halo_h <= (int32_t) ih &&
                           (uint32_t) (sx0 / 2) +
                               (tile_w / HTP_CONV2D_TILE_W - 1u) *
                               (HTP_CONV2D_TILE_W / 2u) +
                               (src->type == HTP_TYPE_F16 ? 64u : 32u) <= src->ne[0],
        .input_f16 = src->type == HTP_TYPE_F16,
    };
}

static void conv2d_prepare_activation(
        struct htp_ops_context * octx,
        float * x,
        __fp16 * a,
        uint32_t x0,
        uint32_t y0,
        uint32_t tile_w,
        uint32_t tile_h,
        bool wait_weight_dma,
        const struct htp_conv2d_kernel_params * kp) {
    struct conv2d_prepare_state state;
    conv2d_init_prepare_state(octx, x, a, x0, y0, tile_w, tile_h,
                              wait_weight_dma, kp, &state);
    if (kp->ic != kp->ic_padded) {
        const size_t bytes = (size_t) state.x_tiles * state.halo_h *
                             state.activation_k_tiles * HTP_CONV2D_TILE_BYTES;
        memset(a, 0, bytes);
    }
    work_queue_run(octx->ctx->work_queue, conv2d_prepare_activation_worker, &state, octx->n_threads);
}

struct conv3d_weight_pack_state {
    const __fp16 * src;
    __fp16 * dst;
    uint32_t kw;
    uint32_t kh;
    uint32_t kd;
    uint32_t ic;
    uint32_t ic_padded;
    uint32_t total_oc;
    uint32_t oc_start;
    uint32_t oc_count;
};

static void conv3d_pack_current_slice_worker(unsigned int nth, unsigned int ith, void * data) {
    const struct conv3d_weight_pack_state * st = (const struct conv3d_weight_pack_state *) data;
    const uint32_t area = st->kw * st->kh;
    const uint32_t ic_blocks = st->ic_padded / 32u;
    const uint32_t k_tiles = area * ic_blocks;
    const uint32_t pairs = (st->oc_count + 1u) / 2u;
    const uint32_t first_pair = (uint64_t) pairs * ith / nth;
    const uint32_t last_pair = (uint64_t) pairs * (ith + 1u) / nth;

    for (uint32_t pair = first_pair; pair < last_pair; ++pair) {
        const uint32_t local_oc0 = pair * 2u;
        const uint32_t local_oc1 = local_oc0 + 1u;
        const uint32_t oc0 = st->oc_start + local_oc0;
        const uint32_t oc1 = st->oc_start + local_oc1;
        const uint32_t nt = local_oc0 / 32u;
        const uint32_t nr_pair = (local_oc0 % 32u) / 2u;
        for (uint32_t s = 0; s < area; ++s) {
            for (uint32_t ic = 0; ic < st->ic; ++ic) {
                const uint32_t kt = s * ic_blocks + ic / 32u;
                const uint32_t kr = ic % 32u;
                const size_t tile = ((size_t) nt * k_tiles + kt) *
                                    HTP_MM_HMX_TILE_N_ELMS;
                const size_t dst_index = tile + nr_pair * 64u + kr * 2u;
                const size_t src0_index = s + (size_t) area *
                    ((st->kd - 1u) + (size_t) st->kd * (ic + (size_t) st->ic * oc0));
                st->dst[dst_index] = st->src[src0_index];
                if (oc1 < st->total_oc && local_oc1 < st->oc_count) {
                    const size_t src1_index = s + (size_t) area *
                        ((st->kd - 1u) + (size_t) st->kd * (ic + (size_t) st->ic * oc1));
                    st->dst[dst_index + 1u] = st->src[src1_index];
                }
            }
        }
    }
}

static void conv3d_pack_current_slice(
        struct htp_ops_context * octx,
        const struct htp_conv2d_kernel_params * kp,
        uint32_t oc_start,
        uint32_t oc_count,
        __fp16 * dst) {
    const __fp16 * src = (const __fp16 *) octx->src[0]->data;
    const uint32_t n_tiles = (oc_count + 31u) / 32u;
    memset(dst, 0, (size_t) kp->k_tiles * n_tiles * HTP_CONV2D_TILE_BYTES);
    struct conv3d_weight_pack_state state = {
        .src = src,
        .dst = dst,
        .kw = kp->kw,
        .kh = kp->kh,
        .kd = kp->kd,
        .ic = kp->ic,
        .ic_padded = kp->ic_padded,
        .total_oc = kp->oc,
        .oc_start = oc_start,
        .oc_count = oc_count,
    };
    work_queue_run(octx->ctx->work_queue, conv3d_pack_current_slice_worker,
                   &state, octx->n_threads);
}

struct conv2d_pipeline_state {
    struct conv2d_prepare_state prepare;
    struct conv2d_store_state store;
    struct conv2d_hmx_job * next_job;
    atomic_uint prepare_barrier;
    atomic_uint next_store_pair;
    atomic_bool hmx_ready;
    uint32_t block;
    bool has_next;
};

static void conv2d_pipeline_worker(unsigned int nth, unsigned int ith, void * data) {
    struct conv2d_pipeline_state * st = (struct conv2d_pipeline_state *) data;
    if (st->has_next) {
        conv2d_prepare_activation_worker(nth, ith, &st->prepare);
    }

    atomic_fetch_sub_explicit(&st->prepare_barrier, 1, memory_order_release);
    if (ith == 0) {
        hmx_queue_pop(st->store.ctx->hmx_queue);
        htp_trace_event_start(&st->store.ctx->trace[0], HTP_TRACE_EVT_HVX_O_PROC, st->block);
        atomic_store_explicit(&st->hmx_ready, true, memory_order_release);

        if (st->has_next) {
            while (atomic_load_explicit(&st->prepare_barrier, memory_order_acquire) != 0) {
                hex_pause();
            }
            htp_trace_event_stop(&st->store.ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, st->block + 1);
            hmx_queue_push(st->store.ctx->hmx_queue,
                           hmx_queue_make_desc(conv2d_hmx_worker, st->next_job));
        }
    } else {
        while (!atomic_load_explicit(&st->hmx_ready, memory_order_acquire)) {
            hex_pause();
        }
    }

    conv2d_store_output_worker(nth, ith, &st->store);
}

int op_conv2d(struct htp_ops_context * octx) {
    const struct htp_tensor * weight = octx->src[0];
    const struct htp_tensor * src = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const struct htp_conv2d_kernel_params * kp =
        (const struct htp_conv2d_kernel_params *) octx->kernel_params;

    const bool causal3d = (kp->flags & HTP_CONV2D_CAUSAL3D) != 0;
    if (weight->type != HTP_TYPE_F16 ||
        (src->type != HTP_TYPE_F32 && src->type != HTP_TYPE_F16) ||
        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16) ||
        (src->type == HTP_TYPE_F16 && dst->type != HTP_TYPE_F16) ||
        kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((!causal3d && (src->ne[2] != kp->ic || src->ne[3] != 1 ||
                       dst->ne[2] != kp->oc || dst->ne[3] != 1)) ||
        (causal3d && (src->ne[2] != 1 || src->ne[3] != kp->ic ||
                      dst->ne[2] != 1 || dst->ne[3] != kp->oc ||
                      weight->ne[0] != kp->kw || weight->ne[1] != kp->kh ||
                      weight->ne[2] != kp->kd ||
                      weight->ne[3] != kp->ic * kp->oc))) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_BIAS) &&
        (!octx->src[2] || octx->src[2]->type != HTP_TYPE_F32 ||
         (uint64_t) octx->src[2]->ne[0] * octx->src[2]->ne[1] *
         octx->src[2]->ne[2] * octx->src[2]->ne[3] != kp->oc)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_RESIDUAL) &&
        (!octx->src[3] ||
         octx->src[3]->ne[0] != dst->ne[0] || octx->src[3]->ne[1] != dst->ne[1] ||
         octx->src[3]->ne[2] != dst->ne[2] || octx->src[3]->ne[3] != dst->ne[3] ||
         octx->src[3]->type != dst->type)) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_DUP_UP_RESIDUAL) &&
        (!octx->src[3] || octx->src[3]->type != HTP_TYPE_F32 ||
         src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
         octx->src[3]->ne[0] * 2u != dst->ne[0] ||
         octx->src[3]->ne[1] * 2u != dst->ne[1] ||
         octx->src[3]->ne[2] != 1 ||
         !((kp->residual_factor_t == 2 &&
            (octx->src[3]->ne[3] == kp->oc || octx->src[3]->ne[3] == 2u * kp->oc)) ||
           (kp->residual_factor_t == 1 && octx->src[3]->ne[3] == 2u * kp->oc)))) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if ((kp->flags & HTP_CONV2D_UPSCALE2) &&
        (dst->ne[0] != 2u * src->ne[0] || dst->ne[1] != 2u * src->ne[1] ||
         dst->ne[2] != weight->ne[3] || src->ne[2] != weight->ne[2])) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t oc = kp->oc;
    const uint32_t ow = dst->ne[0];
    const uint32_t oh = dst->ne[1];
    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;
    __fp16 * bbuf[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_weight[0]),
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_weight[1]),
    };
    float * xbuf = VTCM_LAYOUT_PTR(float, base, kp->off_x);
    __fp16 * abuf[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_a[0]),
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_a[1]),
    };
    __fp16 * cbuf[2] = {
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_c[0]),
        VTCM_LAYOUT_PTR(__fp16, base, kp->off_c[1]),
    };
    __fp16 * scales = VTCM_LAYOUT_PTR(__fp16, base, kp->off_scales);

    dma_queue * q = octx->ctx->dma[0];
    const bool causal3d_prepacked = causal3d && (weight->flags & HTP_TENSOR_CONV2D);
    hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));

    const uint32_t nx = (ow + kp->tile_w - 1) / kp->tile_w;
    const uint32_t ny = (oh + kp->tile_h - 1) / kp->tile_h;
    const uint32_t nblocks = nx * ny;
    const float * bias = (kp->flags & HTP_CONV2D_BIAS)
                       ? (const float *) octx->src[2]->data : NULL;
    const float * residual = (kp->flags & (HTP_CONV2D_RESIDUAL |
                                           HTP_CONV2D_DUP_UP_RESIDUAL))
                           ? (const float *) octx->src[3]->data : NULL;
    const uint32_t residual_w = residual ? octx->src[3]->ne[0] : 0;
    const uint32_t residual_h = residual ? octx->src[3]->ne[1] : 0;
    const uint32_t residual_channel_ratio =
        (kp->flags & HTP_CONV2D_DUP_UP_RESIDUAL)
            ? octx->src[3]->ne[3] / kp->oc : 0;
    const uint32_t n_tiles_per_block = kp->n_tiles_per_block;
    const struct conv2d_store_context store_ctx = {
        .ctx = octx->ctx,
        .dst = (float *) dst->data,
        .bias = bias,
        .residual = residual,
        .output_plane = (size_t) ow * oh,
        .ow = ow,
        .oh = oh,
        .residual_w = residual_w,
        .residual_h = residual_h,
        .residual_channel_ratio = residual_channel_ratio,
        .residual_factor_t = kp->residual_factor_t,
        .flags = kp->flags,
        .output_f16 = dst->type == HTP_TYPE_F16,
    };

    if (n_tiles_per_block < kp->n_tiles) {
        const uint32_t n_chunks = (kp->n_tiles + n_tiles_per_block - 1u) /
                                  n_tiles_per_block;
        for (uint32_t block = 0; block < nblocks; ++block) {
            const struct conv2d_tile tile = conv2d_get_tile(kp, ow, oh, nx, block);
            struct conv2d_hmx_job jobs[2];

            const uint32_t first_n_tiles = hex_smin(n_tiles_per_block, kp->n_tiles);
            const uint32_t first_oc_count = hex_smin(oc, first_n_tiles * 32u);
            if (causal3d && !causal3d_prepacked) {
                conv3d_pack_current_slice(octx, kp, 0, first_oc_count, bbuf[0]);
            } else {
                if (!dma_queue_push(q,
                        dma_make_ptr(bbuf[0], (const void *) (uintptr_t) weight->data),
                        HTP_CONV2D_TILE_BYTES, HTP_CONV2D_TILE_BYTES,
                        HTP_CONV2D_TILE_BYTES, kp->k_tiles * first_n_tiles)) {
                    return HTP_STATUS_INTERNAL_ERR;
                }
            }

            htp_trace_event_start(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, block);
            conv2d_prepare_activation(octx, xbuf, abuf[0], tile.x0, tile.y0,
                                      tile.tile_w, tile.tile_h,
                                      !causal3d || causal3d_prepacked, kp);
            htp_trace_event_stop(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, block);

            jobs[0] = conv2d_make_hmx_job(
                cbuf[0], bbuf[0], abuf[0], scales, first_n_tiles, &tile, kp);
            hmx_queue_push(octx->ctx->hmx_queue,
                           hmx_queue_make_desc(conv2d_hmx_worker, &jobs[0]));

            for (uint32_t chunk = 0; chunk < n_chunks; ++chunk) {
                const uint32_t slot = chunk & 1u;
                const uint32_t n_tile_first = chunk * n_tiles_per_block;
                const uint32_t n_tiles = hex_smin(n_tiles_per_block,
                                                   kp->n_tiles - n_tile_first);
                const uint32_t oc_first = n_tile_first * 32u;
                const uint32_t oc_count = hex_smin(oc - oc_first, n_tiles * 32u);
                const bool has_next = chunk + 1u < n_chunks;

                if (has_next) {
                    const uint32_t next = chunk + 1u;
                    const uint32_t next_slot = next & 1u;
                    const uint32_t next_n_tile_first = next * n_tiles_per_block;
                    const uint32_t next_n_tiles = hex_smin(
                        n_tiles_per_block, kp->n_tiles - next_n_tile_first);
                    const uint32_t next_oc_first = next_n_tile_first * 32u;
                    const uint32_t next_oc_count = hex_smin(
                        oc - next_oc_first, next_n_tiles * 32u);
                    if (causal3d && !causal3d_prepacked) {
                        conv3d_pack_current_slice(octx, kp, next_oc_first,
                                                  next_oc_count, bbuf[next_slot]);
                    } else {
                        const uint8_t * weight_src =
                            (const uint8_t *) (uintptr_t) weight->data +
                            (size_t) next_n_tile_first * kp->k_tiles *
                            HTP_CONV2D_TILE_BYTES;
                        if (!dma_queue_push(q,
                                dma_make_ptr(bbuf[next_slot], weight_src),
                                HTP_CONV2D_TILE_BYTES, HTP_CONV2D_TILE_BYTES,
                                HTP_CONV2D_TILE_BYTES, kp->k_tiles * next_n_tiles)) {
                            return HTP_STATUS_INTERNAL_ERR;
                        }
                    }

                    jobs[next_slot] = conv2d_make_hmx_job(
                        cbuf[next_slot], bbuf[next_slot], abuf[0], scales,
                        next_n_tiles, &tile, kp);
                }

                hmx_queue_pop(octx->ctx->hmx_queue);
                if (has_next) {
                    if (!causal3d || causal3d_prepacked) {
                        dma_queue_pop(q);
                    }
                    const uint32_t next_slot = (chunk + 1u) & 1u;
                    hmx_queue_push(octx->ctx->hmx_queue,
                                   hmx_queue_make_desc(conv2d_hmx_worker,
                                                       &jobs[next_slot]));
                }

                struct conv2d_store_state store = conv2d_make_store_state(
                    &store_ctx, &tile, cbuf[slot], oc_first, oc_count,
                    block + 1u == nblocks && !has_next);
                atomic_uint next_pair;
                atomic_init(&next_pair, 0);
                store.next_pair = &next_pair;
                htp_trace_event_start(&octx->ctx->trace[0],
                                      HTP_TRACE_EVT_HVX_O_PROC, block);
                work_queue_run(octx->ctx->work_queue,
                               conv2d_store_output_worker, &store,
                               octx->n_threads);
                htp_trace_event_stop(&octx->ctx->trace[0],
                                     HTP_TRACE_EVT_HVX_O_PROC, block);
            }
        }
        return HTP_STATUS_OK;
    }

    const bool weight_dma = !causal3d || causal3d_prepacked;
    if (weight_dma) {
        if (!dma_queue_push(q,
                dma_make_ptr(bbuf[0], (const void *) (uintptr_t) weight->data),
                HTP_CONV2D_TILE_BYTES, HTP_CONV2D_TILE_BYTES,
                HTP_CONV2D_TILE_BYTES, kp->k_tiles * kp->n_tiles)) {
            return HTP_STATUS_INTERNAL_ERR;
        }
    } else {
        conv3d_pack_current_slice(octx, kp, 0, oc, bbuf[0]);
    }

    struct conv2d_hmx_job jobs[2];
    const struct conv2d_tile first_tile = conv2d_get_tile(kp, ow, oh, nx, 0);
    htp_trace_event_start(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, 0);
    conv2d_prepare_activation(octx, xbuf, abuf[0], first_tile.x0, first_tile.y0,
                              first_tile.tile_w, first_tile.tile_h, weight_dma, kp);
    htp_trace_event_stop(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, 0);

    jobs[0] = conv2d_make_hmx_job(
        cbuf[0], bbuf[0], abuf[0], scales, kp->n_tiles, &first_tile, kp);
    hmx_queue_push(octx->ctx->hmx_queue,
                   hmx_queue_make_desc(conv2d_hmx_worker, &jobs[0]));

    for (uint32_t block = 0; block < nblocks; ++block) {
        const uint32_t slot = block & 1u;
        const struct conv2d_tile tile = conv2d_get_tile(kp, ow, oh, nx, block);
        struct conv2d_pipeline_state pipeline = {
            .store = conv2d_make_store_state(
                &store_ctx, &tile, cbuf[slot], 0, oc, block + 1 == nblocks),
            .next_job = NULL,
            .block = block,
            .has_next = block + 1 < nblocks,
        };

        if (pipeline.has_next) {
            const uint32_t next = block + 1;
            const uint32_t next_slot = next & 1u;
            const struct conv2d_tile next_tile = conv2d_get_tile(kp, ow, oh, nx, next);

            htp_trace_event_start(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_A_PREP, next);
            conv2d_init_prepare_state(octx, xbuf, abuf[next_slot],
                                      next_tile.x0, next_tile.y0,
                                      next_tile.tile_w, next_tile.tile_h,
                                      false, kp, &pipeline.prepare);
            jobs[next_slot] = conv2d_make_hmx_job(
                cbuf[next_slot], bbuf[0], abuf[next_slot], scales,
                kp->n_tiles, &next_tile, kp);
            pipeline.next_job = &jobs[next_slot];
        }

        atomic_init(&pipeline.prepare_barrier, octx->n_threads);
        atomic_init(&pipeline.next_store_pair, 0);
        atomic_init(&pipeline.hmx_ready, false);
        pipeline.store.next_pair = &pipeline.next_store_pair;
        work_queue_run(octx->ctx->work_queue, conv2d_pipeline_worker,
                       &pipeline, octx->n_threads);
        htp_trace_event_stop(&octx->ctx->trace[0], HTP_TRACE_EVT_HVX_O_PROC, block);
    }

    return HTP_STATUS_OK;
}
