#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

bool htp_ops_support_op(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);

// Per-thread last offload attempt metadata.
// The fallback reason is set when an offload attempt fails and is cleared after
// the caller consumes it.
const char * ggml_htp_take_last_fallback_reason(void);
uint64_t     ggml_htp_take_last_copy_bytes(void);

#ifdef __cplusplus
}
#endif
