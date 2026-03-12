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
