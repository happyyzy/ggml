#pragma once

#include <stdbool.h>

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API bool ggml_backend_is_htp(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_htp_reg(void);

// buffer type check
bool ggml_backend_buft_is_rpcmem(ggml_backend_buffer_type_t buft);

#define GGML_HTP_ZIMG_ROPE_INTERLEAVED_NAME "htp_zimg_rope_interleaved"
#define GGML_HTP_ZIMG_ROPE_NEOX_NAME        "htp_zimg_rope_neox"
#define GGML_HTP_DIT_QKNORM_ROPE_INTERLEAVED_NAME "htp_dit_qknorm_rope_interleaved"
#define GGML_HTP_DIT_QKNORM_ROPE_NEOX_NAME        "htp_dit_qknorm_rope_neox"
#define GGML_HTP_FLUX_SS_LINEAR2_FUSED_NAME        "htp_flux_ss_linear2_fused"

enum ggml_htp_zimg_rope_flags {
    GGML_HTP_ZIMG_ROPE_FLAG_INTERLEAVED = 1u << 0,
};

enum ggml_htp_dit_qknorm_rope_userdata_layout {
    GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_BITS = 16,
    GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_MASK = (1u << GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_BITS) - 1u,
};

static inline uintptr_t ggml_htp_dit_qknorm_rope_pack_userdata(uint32_t flags, uint32_t theta_start) {
    return ((uintptr_t) theta_start << GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_BITS) |
           ((uintptr_t) flags & (uintptr_t) GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_MASK);
}

static inline uint32_t ggml_htp_dit_qknorm_rope_unpack_flags(uintptr_t userdata) {
    return (uint32_t) (userdata & (uintptr_t) GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_MASK);
}

static inline uint32_t ggml_htp_dit_qknorm_rope_unpack_theta_start(uintptr_t userdata) {
    return (uint32_t) (userdata >> GGML_HTP_DIT_QKNORM_ROPE_USERDATA_FLAG_BITS);
}

enum ggml_htp_flash_attn_flags {
    GGML_HTP_FLASH_ATTN_FLAG_Q_ROW_MAJOR = 1u << 8,
    GGML_HTP_FLASH_ATTN_FLAG_K_ROW_MAJOR = 1u << 9,
};

typedef struct ggml_htp_runtime_options {
    bool enable_stats;
    bool enable_fallback_stats;
    bool contract_strict;
    const char * fallback_csv_path;
} ggml_htp_runtime_options;

// optional process-wide runtime knobs used by the HTP eager offload route.
GGML_BACKEND_API void ggml_backend_htp_apply_runtime_options(const struct ggml_htp_runtime_options * opts);

#ifdef __cplusplus
}
#endif
