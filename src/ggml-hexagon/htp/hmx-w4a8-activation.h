#ifndef HTP_HMX_W4A8_ACTIVATION_H
#define HTP_HMX_W4A8_ACTIVATION_H

#include <float.h>
#include <math.h>
#include <string.h>

#include "dma-queue.h"
#include "hvx-base.h"
#include "hvx-reduce.h"

struct w4a8_activations {
    const float *src;
    const float *tail;
    uint8_t *dst;
    float *scales;
    HVX_Vector *scratch;
    uint32_t rows;
    uint32_t k;
    uint32_t k0;
    dma_queue_t *dma;
    uint32_t group_size;
};

static inline void convrot4(HVX_Vector *a, HVX_Vector *b, HVX_Vector *c, HVX_Vector *d, bool ieee_input) {
    const HVX_Vector ab = ieee_input ? Q6_Vqf32_vadd_VsfVsf(*a, *b) : Q6_Vqf32_vadd_Vqf32Vqf32(*a, *b);
    const HVX_Vector cd = ieee_input ? Q6_Vqf32_vadd_VsfVsf(*c, *d) : Q6_Vqf32_vadd_Vqf32Vqf32(*c, *d);
    const HVX_Vector ab_diff = ieee_input ? Q6_Vqf32_vsub_VsfVsf(*a, *b) : Q6_Vqf32_vsub_Vqf32Vqf32(*a, *b);
    const HVX_Vector cd_diff = ieee_input ? Q6_Vqf32_vsub_VsfVsf(*c, *d) : Q6_Vqf32_vsub_Vqf32Vqf32(*c, *d);
    *a = Q6_Vqf32_vadd_Vqf32Vqf32(ab, cd_diff);
    *b = Q6_Vqf32_vsub_Vqf32Vqf32(ab, cd_diff);
    *c = Q6_Vqf32_vadd_Vqf32Vqf32(cd, ab_diff);
    *d = Q6_Vqf32_vsub_Vqf32Vqf32(cd, ab_diff);
}

static inline void deal_four_words(HVX_Vector *v) {
    const HVX_VectorPair lo = Q6_W_vdeal_VVR(v[1], v[0], -4);
    const HVX_VectorPair hi = Q6_W_vdeal_VVR(v[3], v[2], -4);
    const HVX_VectorPair even = Q6_W_vdeal_VVR(Q6_V_lo_W(hi), Q6_V_lo_W(lo), -4);
    const HVX_VectorPair odd = Q6_W_vdeal_VVR(Q6_V_hi_W(hi), Q6_V_hi_W(lo), -4);
    v[0] = Q6_V_lo_W(even);
    v[1] = Q6_V_lo_W(odd);
    v[2] = Q6_V_hi_W(even);
    v[3] = Q6_V_hi_W(odd);
}

static inline void shuffle_four_words(HVX_Vector *v) {
    const HVX_VectorPair even = Q6_W_vshuff_VVR(v[2], v[0], -4);
    const HVX_VectorPair odd = Q6_W_vshuff_VVR(v[3], v[1], -4);
    const HVX_VectorPair lo = Q6_W_vshuff_VVR(Q6_V_lo_W(odd), Q6_V_lo_W(even), -4);
    const HVX_VectorPair hi = Q6_W_vshuff_VVR(Q6_V_hi_W(odd), Q6_V_hi_W(even), -4);
    v[0] = Q6_V_lo_W(lo);
    v[1] = Q6_V_hi_W(lo);
    v[2] = Q6_V_lo_W(hi);
    v[3] = Q6_V_hi_W(hi);
}

static inline void convrot_low_radices(HVX_Vector *v, uint32_t group_size) {
    // Keep the transposed order until the four U8 rows can be shuffled together.
    deal_four_words(v);
    convrot4(&v[0], &v[1], &v[2], &v[3], group_size < 64);
    deal_four_words(v);
    if (group_size >= 16) {
        convrot4(&v[0], &v[1], &v[2], &v[3], false);
    }
}

static inline void convrot_stride16(HVX_Vector *v, uint32_t group_size) {
    const HVX_VectorPair ab = Q6_W_vdeal_VVR(v[2], v[0], -64);
    const HVX_VectorPair cd = Q6_W_vdeal_VVR(v[3], v[1], -64);
    HVX_Vector a = Q6_V_lo_W(ab), b = Q6_V_hi_W(ab);
    HVX_Vector c = Q6_V_lo_W(cd), d = Q6_V_hi_W(cd);
    if (group_size >= 64) {
        convrot4(&a, &b, &c, &d, group_size == 64);
    }
    v[0] = a;
    v[1] = b;
    v[2] = c;
    v[3] = d;
}

static inline HVX_Vector convrot_block(HVX_Vector *block, HVX_Vector max_abs, uint32_t group_size) {
    // Carry QFloat32 through the rotation and convert once for quantization.
    HVX_Vector v[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) v[i] = block[i];
    if (group_size == 256) {
        convrot4(&v[0], &v[2], &v[4], &v[6], true);
        convrot4(&v[1], &v[3], &v[5], &v[7], true);
    }
    convrot_stride16(v, group_size);
    convrot_stride16(v + 4, group_size);
    convrot_low_radices(v, group_size);
    convrot_low_radices(v + 4, group_size);

    const HVX_Vector abs_mask = Q6_V_vsplat_R(0x7fffffff);
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i) {
        const HVX_Vector value = Q6_Vsf_equals_Vqf32(v[i]);
        block[i] = value;
        max_abs = Q6_Vw_vmax_VwVw(max_abs, Q6_V_vand_VV(value, abs_mask));
    }
    return max_abs;
}

static inline HVX_Vector quantize_u8_words(HVX_Vector value, HVX_Vector inv_scale) {
    const HVX_Vector scaled = hvx_vec_mul_f32_f32(value, inv_scale);
#if __HVX_ARCH__ < 79
    // The vector conversion truncates. Adjust the magnitude to nearest-even.
    const HVX_Vector whole = Q6_Vw_equals_Vsf(scaled);
    const HVX_Vector fraction = hvx_vec_abs_f32(hvx_vec_sub_f32_f32(scaled, Q6_Vsf_equals_Vw(whole)));
    const HVX_Vector one = Q6_V_vsplat_R(1);
    const HVX_Vector threshold = Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(0x3f000000), Q6_V_vand_VV(whole, one));
    const HVX_VectorPred round_up = Q6_Q_vcmp_gt_VwVw(fraction, threshold);
    const HVX_Vector direction = Q6_V_vor_VV(Q6_Vw_vasr_VwR(scaled, 31), one);
    const HVX_Vector rounded = Q6_Vw_condacc_QVwVw(round_up, whole, direction);
    return Q6_Vw_vadd_VwVw(rounded, Q6_V_vsplat_R(128));
#else
    // The offset rounds [-127, 127] to U8 in the low byte of each word.
    const HVX_Vector offset = Q6_V_vsplat_R(0x4b400080);
    return hvx_vec_add_f32_f32(scaled, offset);
#endif
}

static inline void load_activation_rows(dma_queue_t dma, const struct w4a8_activations *p,
                                        HVX_Vector *dst, uint32_t first, unsigned int rows) {
    const size_t row_bytes = p->k * sizeof(float);
    if (!p->tail) {
        dma_queue_push_single_1d(dma, dma_make_ptr(dst, p->src + (size_t)first*p->k), rows*row_bytes);
        return;
    }
    const size_t head_bytes = p->k0 * sizeof(float);
    const size_t tail_bytes = row_bytes - head_bytes;
    dma_queue_push(dma, dma_make_ptr(dst, p->src + (size_t)first*p->k0),
                   row_bytes, head_bytes, head_bytes, rows);
    dma_queue_push(dma, dma_make_ptr((uint8_t *)dst + head_bytes, p->tail + (size_t)first*(p->k-p->k0)),
                   row_bytes, tail_bytes, tail_bytes, rows);
}

static inline __attribute__((always_inline)) void prepare_w4a8_activations_impl(
        unsigned int nth, unsigned int ith, void *arg, uint32_t group_size, bool prefetch) {
    const struct w4a8_activations *p = arg;
    const uint32_t row_groups = (p->rows + 3) / 4;
    const uint32_t begin = row_groups * ith / nth;
    const uint32_t end = row_groups * (ith + 1) / nth;
    const uint32_t vectors = p->k / 32;
    HVX_Vector *scratch = p->scratch + (size_t)ith * (prefetch ? 8 : 4) * vectors;

    dma_queue_t dma = p->dma[ith];
    const size_t row_bytes = p->k*sizeof(float);
    const float scale_factor = 1.0f / (127.0f * sqrtf((float)group_size));
    if (begin == end) {
        return;
    }
    if (prefetch) {
        const unsigned remaining = p->rows - begin*4;
        load_activation_rows(dma, p, scratch, begin*4, remaining < 4 ? remaining : 4);
    }
    for (uint32_t group = begin; group < end; ++group) {
        const unsigned remaining = p->rows - group*4;
        const unsigned valid = remaining < 4 ? remaining : 4;
        HVX_Vector *rotated = scratch + (prefetch && ((group-begin) & 1) ? 4*vectors : 0);
        if (prefetch) {
            dma_queue_flush(dma);
            if (group+1 < end) {
                const unsigned next_rows = p->rows - (group+1)*4;
                HVX_Vector *next = rotated == scratch ? scratch + 4*vectors : scratch;
                load_activation_rows(dma, p, next, (group+1)*4, next_rows < 4 ? next_rows : 4);
            }
        } else {
            load_activation_rows(dma, p, rotated, group*4, 1);
        }
        HVX_Vector inv_scale[4];
        for (unsigned r = 0; r < valid; ++r) {
            HVX_Vector *row = rotated + r*vectors;
            if (!prefetch) {
                dma_queue_flush(dma);
                if (r + 1 < valid) {
                    load_activation_rows(dma, p, row + vectors, group*4 + r + 1, 1);
                }
            }
            HVX_Vector max_abs = Q6_V_vzero();
            for (uint32_t i = 0; i < vectors; i += 8) {
                max_abs = convrot_block(row + i, max_abs, group_size);
            }
            const float amax = hvx_vec_get_f32(hvx_vec_reduce_max_i32(max_abs));
            // Fold the rotation normalization into the per-row scale.
            p->scales[group * 4 + r] = amax * scale_factor;
            inv_scale[r] = hvx_vec_splat_f32(amax >= 128.0f * FLT_MIN ? 127.0f / amax : 0.0f);
        }
        for (unsigned r = valid; r < 4; ++r) {
            memset(rotated + r*vectors, 0, row_bytes);
            p->scales[group * 4 + r] = 0.0f;
            inv_scale[r] = Q6_V_vzero();
        }

        HVX_Vector *dst = (HVX_Vector *)p->dst + (size_t)(group / 16) * 16 * vectors + group % 16;
        for (uint32_t i = 0; i < vectors; i += 4, dst += 64) {
            HVX_Vector packed[4];
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const HVX_Vector a = quantize_u8_words(rotated[i + j], inv_scale[0]);
                const HVX_Vector b = quantize_u8_words(rotated[i + j + vectors], inv_scale[1]);
                const HVX_Vector c = quantize_u8_words(rotated[i + j + 2 * vectors], inv_scale[2]);
                const HVX_Vector d = quantize_u8_words(rotated[i + j + 3 * vectors], inv_scale[3]);
                const HVX_Vector ac = Q6_Vh_vshuffe_VhVh(c, a);
                const HVX_Vector bd = Q6_Vh_vshuffe_VhVh(d, b);
                packed[j] = Q6_Vb_vshuffe_VbVb(bd, ac);
            }
            shuffle_four_words(packed);
            shuffle_four_words(packed);
            // Restore the two 16-column halves of each K32 tile.
            const HVX_VectorPair even = Q6_W_vdeal_VVR(packed[1], packed[0], -64);
            const HVX_VectorPair odd = Q6_W_vdeal_VVR(packed[3], packed[2], -64);
            dst[0] = Q6_V_lo_W(even);
            dst[16] = Q6_V_lo_W(odd);
            dst[32] = Q6_V_hi_W(even);
            dst[48] = Q6_V_hi_W(odd);
        }
    }
}

static void prepare_w4a8_activations(unsigned int nth, unsigned int ith, void *arg) {
    const struct w4a8_activations *p = arg;
    prepare_w4a8_activations_impl(nth, ith, arg, p->group_size, false);
}

static void prepare_w4a8_activations_h256(unsigned int nth, unsigned int ith, void *arg) {
    prepare_w4a8_activations_impl(nth, ith, arg, 256, false);
}

static void prepare_w4a8_activations_h256_prefetch(unsigned int nth, unsigned int ith, void *arg) {
    prepare_w4a8_activations_impl(nth, ith, arg, 256, true);
}

#endif
