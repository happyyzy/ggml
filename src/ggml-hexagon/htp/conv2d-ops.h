#ifndef HTP_CONV2D_OPS_H
#define HTP_CONV2D_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HTP_CONV2D_TILE_W 32u
#define HTP_CONV2D_TILE_BYTES 2048u

enum htp_conv2d_flags {
    HTP_CONV2D_BIAS     = 1u << 0,
    HTP_CONV2D_RESIDUAL = 1u << 1,
    HTP_CONV2D_UPSCALE2 = 1u << 2,
    HTP_CONV2D_CAUSAL3D = 1u << 3,
    HTP_CONV2D_DUP_UP_RESIDUAL = 1u << 4,
};

struct htp_conv2d_kernel_params {
    uint32_t tile_w;
    uint32_t tile_h;
    uint32_t m_tiles;
    uint32_t k_tiles;
    uint32_t activation_k_tiles;
    uint32_t n_tiles;
    uint32_t n_tiles_per_block;
    uint32_t activation_group_channels;

    uint32_t off_weight[2];
    uint32_t off_x;
    uint32_t off_a[2];
    uint32_t off_c[2];
    uint32_t off_scales;
    uint32_t vtcm_size;
    uint32_t flags;
    uint32_t kw;
    uint32_t kh;
    uint32_t kd;
    uint32_t ic;
    uint32_t ic_padded;
    uint32_t oc;
    uint32_t residual_factor_t;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128,
              "htp_conv2d_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128,
               "htp_conv2d_kernel_params is too large for kernel_params blob");
#endif

static inline uint32_t htp_conv2d_align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static inline uint32_t htp_conv2d_layout_build(
        struct htp_conv2d_kernel_params * p,
        uint32_t kh,
        uint32_t kw,
        uint32_t ic,
        uint32_t oc,
        uint32_t n_tiles_per_block,
        uint32_t tile_w,
        uint32_t tile_h,
        uint32_t n_threads,
        uint32_t input_element_size) {
    const uint32_t k = kh * kw * ic;
    const uint32_t halo_w = tile_w + kw - 1u;
    const uint32_t halo_h = tile_h + kh - 1u;
    uint32_t off = 0;

    p->tile_w = tile_w;
    p->flags = 0;
    p->kw = kw;
    p->kh = kh;
    p->kd = 1;
    p->ic = ic;
    p->ic_padded = ic;
    p->oc = oc;
    p->residual_factor_t = 0;
    p->tile_h = tile_h;
    p->m_tiles = (tile_w / HTP_CONV2D_TILE_W) * tile_h;
    p->k_tiles = k / 32u;
    p->activation_k_tiles = kw * ic / 32u;
    p->n_tiles = (oc + 31u) / 32u;
    p->n_tiles_per_block = n_tiles_per_block;

    const uint32_t split_n = p->n_tiles_per_block < p->n_tiles;
    const uint32_t weight_bytes = k * p->n_tiles_per_block * 32u * sizeof(uint16_t);
    const uint32_t weight_slots = split_n ? 2u : 1u;
    for (uint32_t i = 0; i < weight_slots; ++i) {
        p->off_weight[i] = off;
        off = htp_conv2d_align_up(off + weight_bytes, HTP_CONV2D_TILE_BYTES);
    }
    if (!split_n) {
        p->off_weight[1] = p->off_weight[0];
    }

    const uint32_t ic_blocks = ic / 32u;
    uint32_t groups_per_ic_block = 1;
    while (groups_per_ic_block < 16u &&
           ic_blocks * groups_per_ic_block < 2u * n_threads) {
        groups_per_ic_block *= 2u;
    }
    p->activation_group_channels = 32u / groups_per_ic_block;
    const uint32_t n_groups = ic / p->activation_group_channels;
    uint32_t x_slots = 0;
    for (uint32_t i = 0; i < n_threads; ++i) {
        const uint32_t first = (uint64_t) n_groups * i / n_threads;
        const uint32_t last = (uint64_t) n_groups * (i + 1u) / n_threads;
        const uint32_t count = last - first;
        x_slots += count < 2u ? count : 2u;
    }
    const uint32_t x_slot_bytes = x_slots * p->activation_group_channels *
                                  halo_w * halo_h * input_element_size;
    p->off_x = off;
    off = htp_conv2d_align_up(off + x_slot_bytes, HTP_CONV2D_TILE_BYTES);

    const uint32_t a_slot_bytes = HTP_CONV2D_TILE_W * (tile_w / HTP_CONV2D_TILE_W) *
                                  halo_h * (kw * ic) * sizeof(uint16_t);
    const uint32_t activation_slots = split_n ? 1u : 2u;
    for (uint32_t i = 0; i < activation_slots; ++i) {
        p->off_a[i] = off;
        off = htp_conv2d_align_up(off + a_slot_bytes, HTP_CONV2D_TILE_BYTES);
    }
    if (split_n) {
        p->off_a[1] = p->off_a[0];
    }

    const uint32_t c_slot_bytes = HTP_CONV2D_TILE_W * p->m_tiles *
                                  p->n_tiles_per_block * 32u * sizeof(uint16_t);
    for (uint32_t i = 0; i < 2; ++i) {
        p->off_c[i] = off;
        off = htp_conv2d_align_up(off + c_slot_bytes, HTP_CONV2D_TILE_BYTES);
    }

    p->off_scales = off;
    off = htp_conv2d_align_up(off + 256u, HTP_CONV2D_TILE_BYTES);

    p->vtcm_size = off;
    return off;
}

#ifdef __cplusplus
}
#endif

#endif
