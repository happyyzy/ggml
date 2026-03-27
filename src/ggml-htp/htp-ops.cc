#include "htp-ops.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <inttypes.h>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "dsprpc_interface.h"
#include "ggml-htp-impl.h"
#include "ggml-htp.h"
#include "ggml.h"

////// Special headers intended for CPU-NPU communication. Keep them in sync with ops backend.
#include "message.h"
#include "op_reg.h"

namespace {

struct HtpOpStats {
    bool enabled = false;
    std::atomic<uint64_t> offload_total{ 0 };
    std::atomic<uint64_t> offload_mul_mat{ 0 };
    std::atomic<uint64_t> offload_flash_attn{ 0 };
    std::atomic<uint64_t> offload_other{ 0 };
    std::atomic<uint64_t> offload_us{ 0 };
    std::atomic<uint64_t> unsupported_mul_mat{ 0 };
    std::atomic<uint64_t> flash_seen{ 0 };
    std::atomic<uint64_t> flash_mask_null{ 0 };
    std::atomic<uint64_t> flash_type_mismatch{ 0 };
};

enum class HtpFallbackReason : uint32_t {
    kSkipByEnv = 0,
    kBackendUninitialized,
    kRmsNormDisabled,
    kUnknownOp,
    kMatmulNameFilter,
    kMatmulAdalnDisabled,
    kMatmulMinShape,
    kMatmulContiguousOrAlign,
    kMatmulF16Disabled,
    kMatmulF16Shape,
    kMatmulQuantDisabled,
    kMatmulQ4Shape,
    kMatmulQ8Shape,
    kMatmulIQ4Shape,
    kMatmulTypeUnsupported,
    kFlashDisabled,
    kFlashContiguousOrAlign,
    kFlashTypeContract,
    kComputeFailureSingleThread,
    kCount,
};

constexpr size_t HTP_FALLBACK_REASON_COUNT =
    static_cast<size_t>(HtpFallbackReason::kCount);

struct HtpFallbackStats {
    bool enabled = false;
    std::array<std::atomic<uint64_t>, HTP_FALLBACK_REASON_COUNT> reason_counts;
    std::array<std::array<std::atomic<uint64_t>, HTP_FALLBACK_REASON_COUNT>, GGML_OP_COUNT> op_reason_counts;
    char csv_path[512] = { 0 };
};

struct HtpPrepackStats {
    bool enabled = false;
    std::atomic<uint64_t> permute_calls{ 0 };
    std::atomic<uint64_t> repack_calls{ 0 };
    std::atomic<uint64_t> permute_bytes{ 0 };
    std::atomic<uint64_t> repack_bytes{ 0 };
    std::atomic<uint64_t> permute_us{ 0 };
    std::atomic<uint64_t> repack_us{ 0 };
};

struct HtpLastAttemptState {
    HtpFallbackReason fallback_reason = HtpFallbackReason::kCount;
    uint64_t copy_bytes = 0;
};

thread_local HtpLastAttemptState g_htp_last_attempt_state{};

void htp_last_attempt_reset() {
    g_htp_last_attempt_state.fallback_reason = HtpFallbackReason::kCount;
    g_htp_last_attempt_state.copy_bytes = 0;
}

// Matmul tracing is opt-in and designed to be grep-friendly and low-overhead.
// It is used to answer: which matmul fast-path is hit under the "required-path"
// (full-offload + flash on + split_m=0) for diffusion vs LLM workloads.
//
// Usage:
//   GGML_HTP_TRACE_MATMUL=1 GGML_HTP_TRACE_MATMUL_MIN_M=128 GGML_HTP_TRACE_MATMUL_MAX=200 ./...
static int g_htp_trace_matmul = -1;
static int g_htp_trace_matmul_min_m = 0;
static int g_htp_trace_matmul_max = 0;
static std::atomic<int> g_htp_trace_matmul_emitted{0};

static inline void htp_trace_matmul_init_once() {
    if (g_htp_trace_matmul >= 0) {
        return;
    }

    const char * v = std::getenv("GGML_HTP_TRACE_MATMUL");
    g_htp_trace_matmul = (v && v[0]) ? std::atoi(v) : 0;

    if (!g_htp_trace_matmul) {
        g_htp_trace_matmul_min_m = 0;
        g_htp_trace_matmul_max = 0;
        return;
    }

    const char * v_min_m = std::getenv("GGML_HTP_TRACE_MATMUL_MIN_M");
    g_htp_trace_matmul_min_m = (v_min_m && v_min_m[0]) ? std::atoi(v_min_m) : 0;

    const char * v_max = std::getenv("GGML_HTP_TRACE_MATMUL_MAX");
    // Default cap prevents accidental log floods on large graphs.
    g_htp_trace_matmul_max = (v_max && v_max[0]) ? std::atoi(v_max) : 200;
}

static inline bool htp_trace_matmul_should_log(int m) {
    htp_trace_matmul_init_once();
    if (!g_htp_trace_matmul) {
        return false;
    }
    if (m < g_htp_trace_matmul_min_m) {
        return false;
    }
    const int idx = g_htp_trace_matmul_emitted.fetch_add(1, std::memory_order_relaxed);
    return g_htp_trace_matmul_max <= 0 ? true : (idx < g_htp_trace_matmul_max);
}

static inline const char * htp_matmul_op_name(int op_index) {
    switch (op_index) {
        case HTP_OPS_MAT_MUL_PERMUTED_W16A32:          return "W16A32_PERMUTED";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32:        return "W4D16A32_PERMUTED";
        case HTP_OPS_MAT_MUL_PERMUTED_W8D16A32:        return "W8D16A32_PERMUTED";
        case HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL: return "W4D16A32_IQ4_NL_PERMUTED";
        case HTP_OPS_MAT_MUL_COMMON_W4D16A32:          return "W4D16A32_COMMON";
        case HTP_OPS_MAT_MUL_COMMON_W8D16A32:          return "W8D16A32_COMMON";
        case HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL:   return "W4D16A32_IQ4_NL_COMMON";
        case HTP_OPS_FLUX_SS_LINEAR2_FUSED_Q8:         return "FLUX_SS_LINEAR2_FUSED_Q8";
        default:                                      return "UNKNOWN";
    }
}

uint64_t htp_estimate_copy_bytes(const ggml_tensor * dst) {
    if (dst == nullptr) {
        return 0;
    }
    uint64_t total = ggml_nbytes(dst);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const ggml_tensor * src = dst->src[i];
        if (src == nullptr) {
            continue;
        }
        total += ggml_nbytes(src);
    }
    return total;
}

void htp_op_stats_dump();
void htp_fallback_stats_dump();
void htp_fallback_record(HtpFallbackReason reason, const ggml_tensor * dst);
void htp_prepack_stats_dump();

static inline bool htp_is_zimg_rope_op(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_MAP_CUSTOM2) {
        return false;
    }
    return std::strcmp(dst->name, GGML_HTP_ZIMG_ROPE_INTERLEAVED_NAME) == 0 ||
           std::strcmp(dst->name, GGML_HTP_ZIMG_ROPE_NEOX_NAME) == 0;
}

static inline bool htp_is_zimg_qknorm_rope_op(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_MAP_CUSTOM3) {
        return false;
    }
    return std::strcmp(dst->name, GGML_HTP_ZIMG_QKNORM_ROPE_INTERLEAVED_NAME) == 0 ||
           std::strcmp(dst->name, GGML_HTP_ZIMG_QKNORM_ROPE_NEOX_NAME) == 0;
}

static inline bool htp_is_flux_ss_linear2_fused_op(const ggml_tensor * dst) {
    return dst != nullptr &&
           dst->op == GGML_OP_MAP_CUSTOM3 &&
           std::strcmp(dst->name, GGML_HTP_FLUX_SS_LINEAR2_FUSED_NAME) == 0;
}

static inline uint32_t htp_zimg_rope_flags(const ggml_tensor * dst) {
    return std::strcmp(dst->name, GGML_HTP_ZIMG_ROPE_INTERLEAVED_NAME) == 0 ? HTP_ZIMG_ROPE_FLAG_INTERLEAVED : 0u;
}

static inline uint32_t htp_zimg_qknorm_rope_flags(const ggml_tensor * dst) {
    return std::strcmp(dst->name, GGML_HTP_ZIMG_QKNORM_ROPE_INTERLEAVED_NAME) == 0 ? HTP_ZIMG_ROPE_FLAG_INTERLEAVED : 0u;
}

static inline uint32_t htp_zimg_qknorm_rope_theta_start(const ggml_tensor * dst) {
    struct ggml_map_custom3_op_params params;
    std::memcpy(&params, dst->op_params, sizeof(params));
    return ggml_htp_zimg_qknorm_rope_unpack_theta_start(reinterpret_cast<uintptr_t>(params.userdata));
}

static inline bool htp_zimg_rope_contract_ok(const ggml_tensor * dst) {
    if (!htp_is_zimg_rope_op(dst)) {
        return false;
    }

    const ggml_tensor * src   = dst->src[0];
    const ggml_tensor * theta = dst->src[1];
    if (src == nullptr || theta == nullptr) {
        return false;
    }

    constexpr size_t kVecAlign = 128;
    auto ptr_aligned = [](const void * ptr, size_t align) {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
    };
    const bool type_ok = dst->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32 && theta->type == GGML_TYPE_F32;
    const bool shape_ok = dst->ne[0] > 0 && (dst->ne[0] % 2) == 0 &&
                          src->ne[0] == dst->ne[0] && src->ne[1] == dst->ne[1] &&
                          src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3] &&
                          theta->ne[0] == 2 && theta->ne[1] == 2 &&
                          theta->ne[2] * 2 == dst->ne[0] &&
                          theta->ne[3] == dst->ne[1];
    const bool contiguous_ok = ggml_is_contiguous(dst) && ggml_is_contiguous(src) && ggml_is_contiguous(theta);
    const bool aligned_ok = ptr_aligned(dst->data, kVecAlign) &&
                            ptr_aligned(src->data, kVecAlign) &&
                            ptr_aligned(theta->data, kVecAlign);

    return type_ok && shape_ok && contiguous_ok && aligned_ok;
}

static inline bool htp_zimg_qknorm_rope_contract_ok(const ggml_tensor * dst) {
    if (!htp_is_zimg_qknorm_rope_op(dst)) {
        return false;
    }

    const ggml_tensor * src    = dst->src[0];
    const ggml_tensor * weight = dst->src[1];
    const ggml_tensor * theta  = dst->src[2];
    if (src == nullptr || weight == nullptr || theta == nullptr) {
        return false;
    }
    const uint32_t theta_start = htp_zimg_qknorm_rope_theta_start(dst);

    constexpr size_t kVecAlign = 128;
    auto ptr_aligned = [](const void * ptr, size_t align) {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
    };
    auto stride_mul_float = [](size_t stride) {
        return (stride % sizeof(float)) == 0;
    };

    const bool type_ok = dst->type == GGML_TYPE_F32 &&
                         src->type == GGML_TYPE_F32 &&
                         weight->type == GGML_TYPE_F32 &&
                         theta->type == GGML_TYPE_F32;
    const bool shape_ok = dst->ne[0] > 0 && (dst->ne[0] % 2) == 0 &&
                          src->ne[0] == dst->ne[0] &&
                          src->ne[1] == dst->ne[1] &&
                          src->ne[2] == dst->ne[2] &&
                          src->ne[3] == dst->ne[3] &&
                          weight->ne[0] == dst->ne[0] &&
                          weight->ne[1] == 1 &&
                          weight->ne[2] * weight->ne[3] == dst->ne[2] * dst->ne[3] &&
                          theta->ne[0] == 2 && theta->ne[1] == 2 &&
                          theta->ne[2] * 2 == dst->ne[0] &&
                          theta->ne[3] >= (int64_t) theta_start + dst->ne[1];
    const bool src_layout_ok = src->nb[0] == sizeof(float) &&
                               stride_mul_float(src->nb[1]) &&
                               stride_mul_float(src->nb[2]) &&
                               stride_mul_float(src->nb[3]) &&
                               src->nb[1] >= src->nb[0] * src->ne[0] &&
                               src->nb[2] >= src->nb[0] * src->ne[0] &&
                               src->nb[3] >= src->nb[2] * src->ne[2];
    const bool contiguous_ok = ggml_is_contiguous(dst) &&
                               src_layout_ok &&
                               ggml_is_contiguous(weight) &&
                               ggml_is_contiguous(theta);
    const bool aligned_ok = ptr_aligned(dst->data, kVecAlign) &&
                            ptr_aligned(src->data, kVecAlign) &&
                            ptr_aligned(weight->data, kVecAlign) &&
                            ptr_aligned(theta->data, kVecAlign);

    return type_ok && shape_ok && contiguous_ok && aligned_ok;
}

HtpOpStats & htp_op_stats() {
    static HtpOpStats stats;
    static bool initialized = false;
    static bool registered = false;
    if (!initialized) {
        const char * env = std::getenv("GGML_HTP_STATS");
        stats.enabled    = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
        initialized      = true;
    }
    if (stats.enabled && !registered) {
        std::atexit(htp_op_stats_dump);
        registered = true;
    }
    return stats;
}

HtpPrepackStats & htp_prepack_stats() {
    static HtpPrepackStats stats;
    static bool initialized = false;
    static bool registered = false;
    if (!initialized) {
        const char * env_enabled = std::getenv("GGML_HTP_PREPACK_STATS");
        if (env_enabled && env_enabled[0] != '\0' && std::strcmp(env_enabled, "0") != 0) {
            stats.enabled = true;
        } else {
            const char * env_stats = std::getenv("GGML_HTP_STATS");
            stats.enabled = env_stats && env_stats[0] != '\0' && std::strcmp(env_stats, "0") != 0;
        }
        initialized = true;
    }
    if (stats.enabled && !registered) {
        std::atexit(htp_prepack_stats_dump);
        registered = true;
    }
    return stats;
}

void htp_prepack_stats_record(bool is_repack, uint64_t bytes, uint64_t us) {
    auto & stats = htp_prepack_stats();
    if (!stats.enabled) {
        return;
    }
    if (is_repack) {
        stats.repack_calls.fetch_add(1, std::memory_order_relaxed);
        stats.repack_bytes.fetch_add(bytes, std::memory_order_relaxed);
        stats.repack_us.fetch_add(us, std::memory_order_relaxed);
    } else {
        stats.permute_calls.fetch_add(1, std::memory_order_relaxed);
        stats.permute_bytes.fetch_add(bytes, std::memory_order_relaxed);
        stats.permute_us.fetch_add(us, std::memory_order_relaxed);
    }
}

const char * htp_fallback_reason_name(HtpFallbackReason reason) {
    switch (reason) {
        case HtpFallbackReason::kSkipByEnv: return "skip-by-env";
        case HtpFallbackReason::kBackendUninitialized: return "backend-uninitialized";
        case HtpFallbackReason::kRmsNormDisabled: return "rms-norm-disabled";
        case HtpFallbackReason::kUnknownOp: return "unknown-op";
        case HtpFallbackReason::kMatmulNameFilter: return "matmul-name-filter";
        case HtpFallbackReason::kMatmulAdalnDisabled: return "matmul-adaln-disabled";
        case HtpFallbackReason::kMatmulMinShape: return "matmul-min-shape";
        case HtpFallbackReason::kMatmulContiguousOrAlign: return "matmul-contiguous-or-align";
        case HtpFallbackReason::kMatmulF16Disabled: return "matmul-f16-disabled";
        case HtpFallbackReason::kMatmulF16Shape: return "matmul-f16-shape";
        case HtpFallbackReason::kMatmulQuantDisabled: return "matmul-quant-disabled";
        case HtpFallbackReason::kMatmulQ4Shape: return "matmul-q4-shape";
        case HtpFallbackReason::kMatmulQ8Shape: return "matmul-q8-shape";
        case HtpFallbackReason::kMatmulIQ4Shape: return "matmul-iq4-shape";
        case HtpFallbackReason::kMatmulTypeUnsupported: return "matmul-type-unsupported";
        case HtpFallbackReason::kFlashDisabled: return "flash-disabled";
        case HtpFallbackReason::kFlashContiguousOrAlign: return "flash-contiguous-or-align";
        case HtpFallbackReason::kFlashTypeContract: return "flash-type-contract";
        case HtpFallbackReason::kComputeFailureSingleThread: return "compute-failed-single-thread";
        case HtpFallbackReason::kCount: return "count";
    }
    return "unknown";
}

HtpFallbackStats & htp_fallback_stats() {
    static HtpFallbackStats stats;
    static bool initialized = false;
    static bool registered = false;
    if (!initialized) {
        const char * env_enabled = std::getenv("GGML_HTP_FALLBACK_STATS");
        if (env_enabled && env_enabled[0] != '\0' && std::strcmp(env_enabled, "0") != 0) {
            stats.enabled = true;
        } else {
            const char * env_stats = std::getenv("GGML_HTP_STATS");
            stats.enabled = env_stats && env_stats[0] != '\0' && std::strcmp(env_stats, "0") != 0;
        }
        const char * env_csv = std::getenv("GGML_HTP_FALLBACK_CSV");
        if (env_csv && env_csv[0] != '\0') {
            std::snprintf(stats.csv_path, sizeof(stats.csv_path), "%s", env_csv);
        } else {
            stats.csv_path[0] = '\0';
        }
        initialized = true;
    }
    if (stats.enabled && !registered) {
        std::atexit(htp_fallback_stats_dump);
        registered = true;
    }
    return stats;
}

void htp_fallback_record(HtpFallbackReason reason, const ggml_tensor * dst) {
    g_htp_last_attempt_state.fallback_reason = reason;

    auto & stats = htp_fallback_stats();
    if (!stats.enabled) {
        return;
    }
    const size_t ridx = static_cast<size_t>(reason);
    if (ridx >= HTP_FALLBACK_REASON_COUNT) {
        return;
    }
    stats.reason_counts[ridx].fetch_add(1, std::memory_order_relaxed);
    if (dst != nullptr && dst->op >= 0 && dst->op < GGML_OP_COUNT) {
        stats.op_reason_counts[dst->op][ridx].fetch_add(1, std::memory_order_relaxed);
    }
}

bool htp_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_DEBUG");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_debug_log_all_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_DEBUG_LOG_ALL_MATMUL");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_runtime_repack_qweights_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_RUNTIME_REPACK_QWEIGHTS");
        // fix1 diffusion GGUFs are already exported in the target prepacked layout.
        // Keep runtime repack disabled by default; explicit opt-in remains available.
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_debug_disable_prepack_cache_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_DEBUG_DISABLE_PREPACK_CACHE");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_runtime_permute_qweights_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS");
        // fix1 diffusion GGUFs are already exported in the target permuted layout.
        // Keep runtime permute disabled by default; explicit opt-in remains available.
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_runtime_permute_f16weights_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_RUNTIME_PERMUTE_F16WEIGHTS");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_f16_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_F16_MATMUL");
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_quant_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_QUANT_MATMUL");
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_adaln_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_ADALN_MATMUL");
        // Keep adaLN matmul off by default; enable only for explicit experiments.
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_cap_embed_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_CAP_EMBED_MATMUL");
        // fix1 validated baseline routes cap_embedder.1 through HMX by default.
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_noise_refiner_w2_matmul_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_NOISE_REFINER_W2_MATMUL");
        // fix1 validated baseline routes noise_refiner.{0,1}.feed_forward.w2 through HMX by default.
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_q8_out_stationary_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_Q8_OUTSTATIONARY");
        // Align with llama.cpp-npu default behavior and allow host-side A/B by
        // toggling a single runtime env on the same binaries.
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_flash_attn_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_ENABLE_FLASH_ATTN");
        enabled = (!env || env[0] == '\0' || std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_flash_force_null_mask_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_FLASH_FORCE_NULL_MASK");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_flash_prepare_in_kernel_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_FLASH_PREP_IN_KERNEL");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_flash_swap_qo_kv_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_FLASH_SWAP_QO_KV");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_matmul_debug_check_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_DEBUG_MATMUL_CHECK");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_matmul_debug_check_all_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_DEBUG_MATMUL_CHECK_ALL");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// Optional filter to reduce GGML_HTP_DEBUG_MATMUL_CHECK overhead.
// If set, only matmul nodes whose weight->name contains the substring will be
// armed for debug metrics/dumps.
const char * htp_matmul_debug_weight_contains_filter() {
    const char * env = std::getenv("GGML_HTP_DEBUG_MATMUL_CHECK_WEIGHT_CONTAINS");
    if (env && env[0] != '\0') {
        return env;
    }
    return nullptr;
}

const char * htp_debug_dump_weight_contains_filter() {
    const char * env = std::getenv("GGML_HTP_DEBUG_DUMP_WEIGHT_CONTAINS");
    if (env && env[0] != '\0') {
        return env;
    }
    return nullptr;
}

const char * htp_debug_dump_weight_dir() {
    const char * env = std::getenv("GGML_HTP_DEBUG_DUMP_WEIGHT_DIR");
    if (env && env[0] != '\0') {
        return env;
    }
    return "/data/local/tmp";
}

bool htp_debug_dump_weight_exit_enabled() {
    const char * env = std::getenv("GGML_HTP_DEBUG_DUMP_WEIGHT_EXIT");
    return (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? true : false;
}

std::string htp_sanitize_filename(const char * s, size_t max_len) {
    std::string out;
    out.reserve(max_len);
    if (!s) {
        return out;
    }
    for (size_t i = 0; s[i] != '\0' && out.size() < max_len; ++i) {
        const unsigned char ch = (unsigned char) s[i];
        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '.' || ch == '_' || ch == '-') {
            out.push_back((char) ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

void htp_debug_dump_weight_if_requested(const ggml_tensor * weight) {
    if (!weight || !weight->data) {
        return;
    }
    const char * filter = htp_debug_dump_weight_contains_filter();
    if (!filter) {
        return;
    }
    const char * w_name = (weight->name[0] != '\0') ? weight->name : nullptr;
    if (!w_name || std::strstr(w_name, filter) == nullptr) {
        return;
    }

    static std::atomic<uint32_t> dumped{ 0 };
    const uint32_t idx = dumped.fetch_add(1, std::memory_order_relaxed) + 1;
    if (idx != 1) {
        return;
    }

    const std::string safe = htp_sanitize_filename(w_name, 120);
    const std::string base = std::string(htp_debug_dump_weight_dir()) + "/htp_weight_dump_" + safe;
    const std::string bin_path = base + ".bin";
    const std::string meta_path = base + ".meta.txt";

    FILE * f = std::fopen(bin_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "HTP debug dump weight failed: open %s\n", bin_path.c_str());
        return;
    }
    const size_t nbytes = ggml_nbytes(weight);
    const size_t wrote = std::fwrite(weight->data, 1, nbytes, f);
    std::fclose(f);

    FILE * meta = std::fopen(meta_path.c_str(), "wb");
    if (meta) {
        std::fprintf(meta, "name=%s\n", w_name);
        std::fprintf(meta, "type=%s\n", ggml_type_name(weight->type));
        std::fprintf(meta, "ne0=%lld ne1=%lld ne2=%lld ne3=%lld\n",
                     (long long) weight->ne[0], (long long) weight->ne[1],
                     (long long) weight->ne[2], (long long) weight->ne[3]);
        std::fprintf(meta, "nbytes=%zu wrote=%zu\n", nbytes, wrote);
        std::fprintf(meta, "GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=%s\n", std::getenv("GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS") ? std::getenv("GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS") : "");
        std::fprintf(meta, "GGML_HTP_RUNTIME_REPACK_QWEIGHTS=%s\n", std::getenv("GGML_HTP_RUNTIME_REPACK_QWEIGHTS") ? std::getenv("GGML_HTP_RUNTIME_REPACK_QWEIGHTS") : "");
        std::fprintf(meta, "GGML_HTP_SKIP_CORE_PERMUTE_REPACK=%s\n", std::getenv("GGML_HTP_SKIP_CORE_PERMUTE_REPACK") ? std::getenv("GGML_HTP_SKIP_CORE_PERMUTE_REPACK") : "");
        std::fclose(meta);
    }

    std::fprintf(stderr,
                 "HTP debug dump weight ok: %s (%zu bytes, wrote=%zu)\n",
                 bin_path.c_str(), nbytes, wrote);

    if (htp_debug_dump_weight_exit_enabled()) {
        std::fprintf(stderr, "HTP debug dump weight: exiting after first dump\n");
        std::fflush(stderr);
        std::exit(0);
    }
}

int htp_env_int_cached(const char * name, int default_value) {
    // Cache env reads per variable to avoid repeated getenv/parse on hot path.
    static std::mutex mtx;
    static std::unordered_map<std::string, int> cache;

    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(name);
    if (it != cache.end()) {
        return it->second;
    }

    int value = default_value;
    if (const char * env = std::getenv(name)) {
        char * end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end != env) {
            if (parsed < INT_MIN) {
                value = INT_MIN;
            } else if (parsed > INT_MAX) {
                value = INT_MAX;
            } else {
                value = (int) parsed;
            }
        }
    }
    cache.emplace(name, value);
    return value;
}

int htp_matmul_min_m() {
    return htp_env_int_cached("GGML_HTP_MATMUL_MIN_M", 0);
}

int htp_matmul_min_k() {
    return htp_env_int_cached("GGML_HTP_MATMUL_MIN_K", 0);
}

int htp_matmul_min_n() {
    return htp_env_int_cached("GGML_HTP_MATMUL_MIN_N", 0);
}

bool htp_qkv_force_act_q8_enabled() {
    return htp_env_int_cached("GGML_HTP_QKV_FORCE_ACT_Q8", 0) != 0;
}

bool htp_qkv_force_act_q8_copy_enabled() {
    return htp_env_int_cached("GGML_HTP_QKV_FORCE_ACT_Q8_COPY", 0) != 0;
}

bool htp_qkv_force_act_q8_ref_enabled() {
    return htp_env_int_cached("GGML_HTP_QKV_FORCE_ACT_Q8_REF", 0) != 0;
}

bool htp_qkv_force_act_q8_copy_f16_enabled() {
    return htp_env_int_cached("GGML_HTP_QKV_FORCE_ACT_Q8_COPY_FP16", 0) != 0;
}

bool htp_qkv_force_act_f16_copy_enabled() {
    return htp_env_int_cached("GGML_HTP_QKV_FORCE_ACT_F16_COPY", 0) != 0;
}

int htp_matmul_split_m() {
    return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M", 0);
}

int htp_matmul_split_m_for_weight(const ggml_tensor * weight) {
    int split_m = htp_matmul_split_m();
    if (!weight) {
        return split_m;
    }
    const char * name = weight->name;
    if (!name || name[0] == '\0') {
        return split_m;
    }
    // Per-family overrides to allow mixed split policy for DiT core ops.
    if (std::strstr(name, ".attention.qkv.weight") != nullptr) {
        return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M_QKV", split_m);
    }
    if (std::strstr(name, ".attention.out.weight") != nullptr) {
        return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M_OUT", split_m);
    }
    if (std::strstr(name, ".feed_forward.w2.weight") != nullptr) {
        return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M_W2", split_m);
    }
    if (std::strstr(name, ".feed_forward.w1.weight") != nullptr ||
        std::strstr(name, ".feed_forward.w3.weight") != nullptr) {
        return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M_W13", split_m);
    }
    if (std::strstr(name, ".adaLN_modulation.") != nullptr) {
        return htp_env_int_cached("GGML_HTP_MATMUL_SPLIT_M_ADALN", split_m);
    }
    return split_m;
}

int htp_flash_dump_max() {
    return htp_env_int_cached("GGML_HTP_FLASH_DUMP_MAX", 0);
}

bool htp_flash_hash_trace_enabled() {
    return htp_env_int_cached("GGML_HTP_FLASH_HASH_TRACE", 0) != 0;
}

const char * htp_flash_dump_dir() {
    const char * env = std::getenv("GGML_HTP_FLASH_DUMP_DIR");
    if (env && env[0] != '\0') {
        return env;
    }
    return "/data/local/tmp";
}

static uint64_t htp_fnv1a64(const void * data, size_t len) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t htp_tensor_hash64(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) {
        return 0ull;
    }
    return htp_fnv1a64(t->data, ggml_nbytes(t));
}

int htp_matmul_debug_dump_call() {
    return htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_DUMP_CALL", 0);
}

const char * htp_matmul_debug_dump_dir() {
    const char * env = std::getenv("GGML_HTP_DEBUG_MATMUL_DUMP_DIR");
    if (env && env[0] != '\0') {
        return env;
    }
    return "/data/local/tmp";
}

bool htp_matmul_debug_raw_dump_only_selected(uint64_t call_idx) {
    const int  raw_dump_call = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_RAW_DUMP_CALL", 0);
    const bool raw_dump_only = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_RAW_DUMP_ONLY", 0) != 0;
    return raw_dump_only && raw_dump_call > 0 && (int) call_idx == raw_dump_call;
}

bool htp_q8_outstationary_firstblock_dump_enabled() {
    return htp_env_int_cached("GGML_HTP_Q8_OUTSTATIONARY_FIRSTBLOCK_DUMP", 0) != 0;
}

bool htp_q8_outstationary_firstblock_dump_selected(uint64_t call_idx) {
    if (!htp_q8_outstationary_firstblock_dump_enabled()) {
        return false;
    }
    const int raw_dump_call = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_RAW_DUMP_CALL", 0);
    return raw_dump_call > 0 && (int) call_idx == raw_dump_call;
}

bool htp_q8_outstationary_acttile_dump_enabled() {
    return htp_env_int_cached("GGML_HTP_Q8_OUTSTATIONARY_ACTTILE_DUMP", 0) != 0;
}

bool htp_q8_outstationary_act_direct_stage_enabled() {
    return htp_env_int_cached("GGML_HTP_Q8_OUTSTATIONARY_ACT_DIRECT_STAGE", 0) != 0;
}

bool htp_q8_outstationary_act_scratch_direct_transfer_enabled() {
    return htp_env_int_cached("GGML_HTP_Q8_OUTSTATIONARY_ACT_SCRATCH_DIRECT_TRANSFER", 0) != 0;
}

bool htp_q8_outstationary_scratch_scalardump_enabled() {
    return htp_env_int_cached("GGML_HTP_Q8_OUTSTATIONARY_SCRATCH_SCALARDUMP", 0) != 0;
}

bool htp_dump_raw_f32_to_file(const std::string & path, const char * name, const float * data,
                              int32_t ne0, int32_t ne1) {
    if (data == nullptr || ne0 <= 0 || ne1 <= 0) {
        return false;
    }

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    const int32_t n_dims   = 2;
    const int32_t name_len = name ? static_cast<int32_t>(std::strlen(name)) : 0;
    const int32_t ttype    = static_cast<int32_t>(GGML_TYPE_F32);
    const int32_t dims[2]  = { ne0, ne1 };

    std::fwrite(&n_dims, sizeof(n_dims), 1, f);
    std::fwrite(&name_len, sizeof(name_len), 1, f);
    std::fwrite(&ttype, sizeof(ttype), 1, f);
    std::fwrite(dims, sizeof(int32_t), 2, f);
    if (name_len > 0) {
        std::fwrite(name, 1, name_len, f);
    }
    const size_t nbytes = (size_t) ne0 * (size_t) ne1 * sizeof(float);
    std::fwrite(data, 1, nbytes, f);
    std::fflush(f);
    std::fclose(f);
    return true;
}

static inline bool htp_flux_ss_linear2_contract_ok(const ggml_tensor * dst) {
    if (!htp_is_flux_ss_linear2_fused_op(dst)) {
        return false;
    }

    const ggml_tensor * attn   = dst->src[0];
    const ggml_tensor * mlp    = dst->src[1];
    const ggml_tensor * weight = dst->src[2];

    if (attn == nullptr || mlp == nullptr || weight == nullptr) {
        return false;
    }

    if (dst->type != GGML_TYPE_F32 || attn->type != GGML_TYPE_F32 || mlp->type != GGML_TYPE_F32 ||
        weight->type != GGML_TYPE_Q8_0) {
        return false;
    }

    if (!ggml_is_contiguous(dst) || !ggml_is_contiguous(attn) || !ggml_is_contiguous(mlp) || !ggml_is_contiguous(weight)) {
        return false;
    }

    constexpr size_t kVecAlign = 128;
    auto ptr_aligned = [](const void * ptr, size_t align) {
        return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) % align) == 0;
    };
    if (!ptr_aligned(dst->data, kVecAlign) || !ptr_aligned(attn->data, kVecAlign) ||
        !ptr_aligned(mlp->data, kVecAlign) || !ptr_aligned(weight->data, kVecAlign)) {
        return false;
    }

    if (dst->ne[1] != attn->ne[1] || dst->ne[2] != attn->ne[2] || dst->ne[3] != attn->ne[3]) {
        return false;
    }
    if (attn->ne[1] != mlp->ne[1] || attn->ne[2] != mlp->ne[2] || attn->ne[3] != mlp->ne[3]) {
        return false;
    }

    const int64_t attn_k = attn->ne[0];
    const int64_t mlp_k  = mlp->ne[0];
    const int64_t n      = dst->ne[0];
    const int64_t total_k = attn_k + mlp_k;

    if (weight->ne[0] != total_k || weight->ne[1] != n) {
        return false;
    }

    if ((attn_k % 32) != 0 || (mlp_k % 32) != 0 || (n % 32) != 0 || (total_k % 32) != 0) {
        return false;
    }

    if (total_k >= 16384) {
        return false;
    }

    return true;
}

bool htp_dump_tensor_to_file(const std::string & path, const char * name, const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    const int32_t n_dims   = ggml_n_dims(tensor);
    const int32_t name_len = name ? static_cast<int32_t>(std::strlen(name)) : 0;
    const int32_t ttype    = static_cast<int32_t>(tensor->type);

    std::fwrite(&n_dims, sizeof(n_dims), 1, f);
    std::fwrite(&name_len, sizeof(name_len), 1, f);
    std::fwrite(&ttype, sizeof(ttype), 1, f);
    for (int i = 0; i < n_dims; ++i) {
        const int32_t ne = static_cast<int32_t>(tensor->ne[i]);
        std::fwrite(&ne, sizeof(ne), 1, f);
    }
    if (name_len > 0) {
        std::fwrite(name, 1, name_len, f);
    }
    const size_t nbytes = ggml_nbytes(tensor);
    std::fwrite(tensor->data, 1, nbytes, f);
    std::fflush(f);
    std::fclose(f);
    return true;
}

bool htp_dump_raw_bytes_tensor_to_file(const std::string & path, const char * name, ggml_type type,
                                       const void * data, size_t nbytes, int32_t ne0, int32_t ne1) {
    if (data == nullptr || nbytes == 0 || ne0 <= 0 || ne1 <= 0) {
        return false;
    }

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    const int32_t n_dims   = 2;
    const int32_t name_len = name ? static_cast<int32_t>(std::strlen(name)) : 0;
    const int32_t ttype    = static_cast<int32_t>(type);
    const int32_t dims[2]  = { ne0, ne1 };

    std::fwrite(&n_dims, sizeof(n_dims), 1, f);
    std::fwrite(&name_len, sizeof(name_len), 1, f);
    std::fwrite(&ttype, sizeof(ttype), 1, f);
    std::fwrite(dims, sizeof(int32_t), 2, f);
    if (name_len > 0) {
        std::fwrite(name, 1, name_len, f);
    }
    std::fwrite(data, 1, nbytes, f);
    std::fflush(f);
    std::fclose(f);
    return true;
}

void htp_flash_dump_tensors(int idx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                            const ggml_tensor * mask, const ggml_tensor * dst, float scale, float max_bias,
                            float logit_softcap, uint32_t flags, float kv_scale) {
    const std::string base = std::string(htp_flash_dump_dir()) + "/htp_flash_" + std::to_string(idx);
    const bool q_ok = htp_dump_tensor_to_file(base + "_q.tensor", "q", q);
    const bool k_ok = htp_dump_tensor_to_file(base + "_k.tensor", "k", k);
    const bool v_ok = htp_dump_tensor_to_file(base + "_v.tensor", "v", v);
    const bool o_ok = htp_dump_tensor_to_file(base + "_o.tensor", "o", dst);
    bool m_ok = true;
    if (mask != nullptr) {
        m_ok = htp_dump_tensor_to_file(base + "_mask.tensor", "mask", mask);
    }

    FILE * meta = std::fopen((base + "_meta.txt").c_str(), "wb");
    if (meta != nullptr) {
        std::fprintf(meta, "idx=%d\n", idx);
        std::fprintf(meta, "scale=%g max_bias=%g logit_softcap=%g\n", scale, max_bias, logit_softcap);
        std::fprintf(meta, "flags=0x%08x kv_scale=%g\n", flags, kv_scale);
        if (q != nullptr) {
            std::fprintf(meta, "q_type=%s q_ne=[%ld,%ld,%ld,%ld]\n",
                         ggml_type_name(q->type), q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
            std::fprintf(meta, "q_nb=[%zu,%zu,%zu,%zu]\n",
                         (size_t) q->nb[0], (size_t) q->nb[1], (size_t) q->nb[2], (size_t) q->nb[3]);
        }
        if (k != nullptr) {
            std::fprintf(meta, "k_type=%s k_ne=[%ld,%ld,%ld,%ld]\n",
                         ggml_type_name(k->type), k->ne[0], k->ne[1], k->ne[2], k->ne[3]);
            std::fprintf(meta, "k_nb=[%zu,%zu,%zu,%zu]\n",
                         (size_t) k->nb[0], (size_t) k->nb[1], (size_t) k->nb[2], (size_t) k->nb[3]);
        }
        if (v != nullptr) {
            std::fprintf(meta, "v_type=%s v_ne=[%ld,%ld,%ld,%ld]\n",
                         ggml_type_name(v->type), v->ne[0], v->ne[1], v->ne[2], v->ne[3]);
            std::fprintf(meta, "v_nb=[%zu,%zu,%zu,%zu]\n",
                         (size_t) v->nb[0], (size_t) v->nb[1], (size_t) v->nb[2], (size_t) v->nb[3]);
        }
        if (mask != nullptr) {
            std::fprintf(meta, "mask_type=%s mask_ne=[%ld,%ld,%ld,%ld]\n",
                         ggml_type_name(mask->type), mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3]);
            std::fprintf(meta, "mask_nb=[%zu,%zu,%zu,%zu]\n",
                         (size_t) mask->nb[0], (size_t) mask->nb[1], (size_t) mask->nb[2], (size_t) mask->nb[3]);
        } else {
            std::fprintf(meta, "mask_type=[null]\n");
        }
        if (dst != nullptr) {
            std::fprintf(meta, "o_type=%s o_ne=[%ld,%ld,%ld,%ld]\n",
                         ggml_type_name(dst->type), dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
            std::fprintf(meta, "o_nb=[%zu,%zu,%zu,%zu]\n",
                         (size_t) dst->nb[0], (size_t) dst->nb[1], (size_t) dst->nb[2], (size_t) dst->nb[3]);
        }
        std::fprintf(meta, "dump_ok=q:%d k:%d v:%d mask:%d o:%d\n",
                     q_ok ? 1 : 0, k_ok ? 1 : 0, v_ok ? 1 : 0, m_ok ? 1 : 0, o_ok ? 1 : 0);
        std::fclose(meta);
    }

    if (htp_debug_enabled()) {
        std::fprintf(stderr,
                     "HTP flash dump #%d: q=%d k=%d v=%d mask=%d o=%d base=%s\n",
                     idx, q_ok ? 1 : 0, k_ok ? 1 : 0, v_ok ? 1 : 0, m_ok ? 1 : 0, o_ok ? 1 : 0, base.c_str());
    }
}

std::vector<std::string> htp_parse_csv_env(const char * name) {
    std::vector<std::string> tokens;
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return tokens;
    }
    std::string s(env);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t b = start;
        while (b < end && std::isspace((unsigned char) s[b])) {
            ++b;
        }
        size_t e = end;
        while (e > b && std::isspace((unsigned char) s[e - 1])) {
            --e;
        }
        if (e > b) {
            tokens.emplace_back(s.substr(b, e - b));
        }
        start = end + 1;
    }
    return tokens;
}

bool htp_name_match_any(const char * name, const std::vector<std::string> & tokens) {
    if (!name || tokens.empty()) {
        return false;
    }
    std::string s(name);
    for (const auto & t : tokens) {
        if (!t.empty() && s.find(t) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool htp_matmul_name_allowed(const ggml_tensor * weight) {
    static std::vector<std::string> includes = htp_parse_csv_env("GGML_HTP_MATMUL_NAME_INCLUDE");
    static std::vector<std::string> excludes = htp_parse_csv_env("GGML_HTP_MATMUL_NAME_EXCLUDE");

    const char * name = weight ? weight->name : nullptr;
    if (!includes.empty() && !htp_name_match_any(name, includes)) {
        return false;
    }
    if (!excludes.empty() && htp_name_match_any(name, excludes)) {
        return false;
    }
    return true;
}

bool htp_ptr_aligned(const void * ptr, size_t align) {
    return ptr != nullptr && ((uintptr_t) ptr % align == 0);
}

bool htp_ptr_ranges_overlap(const void * a, size_t a_bytes, const void * b, size_t b_bytes) {
    if (!a || !b || a_bytes == 0 || b_bytes == 0) {
        return false;
    }
    const uintptr_t a0 = (uintptr_t) a;
    const uintptr_t a1 = a0 + a_bytes;
    const uintptr_t b0 = (uintptr_t) b;
    const uintptr_t b1 = b0 + b_bytes;
    return a0 < b1 && b0 < a1;
}

bool htp_skip_prepack_name_from_env(const char * name) {
    static std::vector<std::string> includes = htp_parse_csv_env("GGML_HTP_SKIP_PREPACK_INCLUDE");
    static std::vector<std::string> excludes = htp_parse_csv_env("GGML_HTP_SKIP_PREPACK_EXCLUDE");

    if (!name || name[0] == '\0' || includes.empty()) {
        return false;
    }
    if (!htp_name_match_any(name, includes)) {
        return false;
    }
    if (!excludes.empty() && htp_name_match_any(name, excludes)) {
        return false;
    }
    return true;
}

bool htp_force_permute_name_from_env(const char * name) {
    static std::vector<std::string> includes = htp_parse_csv_env("GGML_HTP_FORCE_PERMUTE_INCLUDE");
    static std::vector<std::string> excludes = htp_parse_csv_env("GGML_HTP_FORCE_PERMUTE_EXCLUDE");

    if (!name || name[0] == '\0' || includes.empty()) {
        return false;
    }
    if (!htp_name_match_any(name, includes)) {
        return false;
    }
    if (!excludes.empty() && htp_name_match_any(name, excludes)) {
        return false;
    }
    return true;
}

bool htp_contract_export_permute_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_CONTRACT_EXPORT_PERMUTE");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_contract_export_repack_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_CONTRACT_EXPORT_REPACK");
        enabled = (env && env[0] != '\0' && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

bool htp_skip_qkv_permute_repack(const ggml_tensor * weight) {
    if (!weight) {
        return false;
    }
    const char * name = weight->name;
    if (htp_skip_prepack_name_from_env(name)) {
        return true;
    }

    // Optional family-specific guards used by diffusion bring-up scans.
    if (htp_env_int_cached("GGML_HTP_SKIP_ADALN_PERMUTE_REPACK", 0) != 0 &&
        std::strstr(name, ".adaLN_modulation.0.weight") != nullptr) {
        return true;
    }
    if (htp_env_int_cached("GGML_HTP_SKIP_CAP_EMBED_PERMUTE_REPACK", 0) != 0 &&
        std::strstr(name, ".cap_embedder.1.weight") != nullptr) {
        return true;
    }

    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_HTP_SKIP_CORE_PERMUTE_REPACK");
        if (env && env[0] != '\0') {
            enabled = (std::strcmp(env, "0") != 0) ? 1 : 0;
        } else {
            // Contract-exported HMX GGUFs already store the core weights in the
            // target permuted/repacked basis. In that case the runtime must use
            // the permuted op family directly instead of routing them through
            // the common-layout path.
            enabled = (htp_contract_export_permute_enabled() || htp_contract_export_repack_enabled()) ? 0 : 1;
        }
    }
    if (!enabled) {
        return false;
    }
    // Task-2 mainline requires W2 q8 out_stationary to stay on the permuted route.
    const bool skip_w2 = htp_env_int_cached("GGML_HTP_SKIP_W2_PERMUTE_REPACK", 0) != 0;
    // For DiT qkv-family in q4/q8 path, runtime float-permute+requant introduces measurable drift.
    // Keep these weights in original quant layout and let DSP side use common dequantization path.
    return std::strstr(name, ".attention.qkv.weight") != nullptr ||
           std::strstr(name, ".attention.q_proj.weight") != nullptr ||
           std::strstr(name, ".attention.k_proj.weight") != nullptr ||
           std::strstr(name, ".attention.v_proj.weight") != nullptr ||
           std::strstr(name, ".attention.out.weight") != nullptr ||
           std::strstr(name, ".feed_forward.w1.weight") != nullptr ||
           (skip_w2 && std::strstr(name, ".feed_forward.w2.weight") != nullptr) ||
           std::strstr(name, ".feed_forward.w3.weight") != nullptr;
}

struct std_block_q4_like {
    uint16_t d;
    uint8_t  qs[16];
} __attribute__((packed));
static_assert(sizeof(std_block_q4_like) == 18, "unexpected q4-like block size");

struct std_block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
} __attribute__((packed));
static_assert(sizeof(std_block_q8_0) == 34, "unexpected q8_0 block size");

struct my_block_q4_0_like {
    uint16_t scales[8];
    uint8_t  quants[8 * 16];
} __attribute__((packed));
static_assert(sizeof(my_block_q4_0_like) == 144, "unexpected packed q4-like super-block size");

struct my_block_q8_0 {
    uint16_t scales[8];
    int8_t   quants[8 * 32];
} __attribute__((packed));
static_assert(sizeof(my_block_q8_0) == 272, "unexpected packed q8_0 super-block size");

static inline float htp_round_to_fp16_scalar(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
}

bool htp_decode_common_qk0_weight_bitexact(ggml_type weight_type,
                                           const uint8_t * raw,
                                           size_t raw_size,
                                           int n_rows,
                                           int k_cols,
                                           std::vector<float> & out) {
    out.clear();
    if (!raw || raw_size == 0 || n_rows <= 0 || k_cols <= 0 || (k_cols % 32) != 0) {
        return false;
    }

    const size_t groups_per_row = (size_t) k_cols / 32;

    if (weight_type == GGML_TYPE_Q4_0 || weight_type == GGML_TYPE_IQ4_NL) {
        const size_t row_bytes = groups_per_row * sizeof(std_block_q4_like);
        const size_t expected  = row_bytes * (size_t) n_rows;
        if (raw_size != expected) {
            return false;
        }

        static const float iq4_nl_table[16] = {
            -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
            1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  113.0f,
        };
        const bool is_iq4_nl = weight_type == GGML_TYPE_IQ4_NL;

        const auto * blocks = reinterpret_cast<const std_block_q4_like *>(raw);
        out.assign((size_t) n_rows * (size_t) k_cols, 0.0f);

        for (int row = 0; row < n_rows; ++row) {
            for (size_t g = 0; g < groups_per_row; ++g) {
                const auto & blk = blocks[(size_t) row * groups_per_row + g];
                const float d = ggml_fp16_to_fp32((ggml_fp16_t) blk.d);
                float * dst = out.data() + (size_t) row * (size_t) k_cols + g * 32;

                for (int i = 0; i < 16; ++i) {
                    const uint8_t q = blk.qs[i];
                    const int ql = q & 0x0F;
                    const int qh = q >> 4;
                    const float vl = is_iq4_nl ? iq4_nl_table[ql] : (float) (ql - 8);
                    const float vh = is_iq4_nl ? iq4_nl_table[qh] : (float) (qh - 8);
                    dst[i]      = htp_round_to_fp16_scalar(vl * d);
                    dst[i + 16] = htp_round_to_fp16_scalar(vh * d);
                }
            }
        }
        return true;
    }

    if (weight_type == GGML_TYPE_Q8_0) {
        const size_t row_bytes = groups_per_row * sizeof(std_block_q8_0);
        const size_t expected  = row_bytes * (size_t) n_rows;
        if (raw_size != expected) {
            return false;
        }

        const auto * blocks = reinterpret_cast<const std_block_q8_0 *>(raw);
        out.assign((size_t) n_rows * (size_t) k_cols, 0.0f);

        for (int row = 0; row < n_rows; ++row) {
            for (size_t g = 0; g < groups_per_row; ++g) {
                const auto & blk = blocks[(size_t) row * groups_per_row + g];
                const float d = ggml_fp16_to_fp32((ggml_fp16_t) blk.d);
                float * dst = out.data() + (size_t) row * (size_t) k_cols + g * 32;
                for (int i = 0; i < 32; ++i) {
                    dst[i] = htp_round_to_fp16_scalar((float) blk.qs[i] * d);
                }
            }
        }
        return true;
    }

    return false;
}

void htp_permute_linear_weight_f32(const float * src, float * dst, int n, int k) {
    const int n_chunks = n / 32;
    const int k_chunks = k / 32;
    const int k_stride = k_chunks * 32;

    for (int r = 0; r < n; ++r) {
        const int64_t row_base = (int64_t) r * k_stride;
        for (int c = 0; c < k_stride; ++c) {
            int64_t t = row_base + c;
            int u     = (int) (t & 1);
            t >>= 1;
            int i = (int) (t & 31);
            t >>= 5;
            int g = (int) (t & 15);
            t >>= 4;
            int b = (int) (t % k_chunks);
            int a = (int) (t / k_chunks);

            if (a < 0 || a >= n_chunks) {
                continue;
            }

            int src_row = a * 32 + i;
            int src_col = b * 32 + g * 2 + u;
            dst[row_base + c] = src[(int64_t) src_row * k + src_col];
        }
    }
}

void repack_row_q4_like_inplace(uint8_t * row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * 8 * sizeof(std_block_q4_like);
        const auto * src_ptr = reinterpret_cast<const std_block_q4_like *>(row + src_off);

        std_block_q4_like src_blocks[8];
        std::memcpy(src_blocks, src_ptr, sizeof(src_blocks));

        my_block_q4_0_like dst_block{};
        uint8_t unpacked_qs[256];
        for (size_t bi = 0; bi < 8; ++bi) {
            dst_block.scales[bi] = src_blocks[bi].d;
            for (size_t j = 0; j < sizeof(src_blocks[bi].qs); ++j) {
                uint8_t q = src_blocks[bi].qs[j];
                // Keep the same nibble permutation as the HMX "packed-quant" layout.
                unpacked_qs[bi * 32 + j + 0]  = q & 0x0F;
                unpacked_qs[bi * 32 + j + 16] = q >> 4;
            }
        }
        for (size_t j = 0; j < sizeof(dst_block.quants) / 2; ++j) {
            // Match llama.cpp-npu HVX repack: interleave (0,128) and (64,192) nibble pairs.
            dst_block.quants[j * 2 + 0] = (uint8_t) ((unpacked_qs[j + 128] << 4) | unpacked_qs[j + 0]);
            dst_block.quants[j * 2 + 1] = (uint8_t) ((unpacked_qs[j + 192] << 4) | unpacked_qs[j + 64]);
        }

        const size_t dst_off = sb * sizeof(my_block_q4_0_like);
        std::memcpy(row + dst_off, &dst_block, sizeof(dst_block));
    }
}

void repack_row_q8_0_inplace(uint8_t * row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * 8 * sizeof(std_block_q8_0);
        const auto * src_ptr = reinterpret_cast<const std_block_q8_0 *>(row + src_off);

        std_block_q8_0 src_blocks[8];
        std::memcpy(src_blocks, src_ptr, sizeof(src_blocks));

        my_block_q8_0 dst_block{};
        for (size_t i = 0; i < 8; ++i) {
            dst_block.scales[i] = src_blocks[i].d;
            std::memcpy(dst_block.quants + i * sizeof(src_blocks[i].qs), src_blocks[i].qs, sizeof(src_blocks[i].qs));
        }

        const size_t dst_off = sb * sizeof(my_block_q8_0);
        std::memcpy(row + dst_off, &dst_block, sizeof(dst_block));
    }
}

bool htp_permute_quant_weight_inplace_if_needed(struct ggml_tensor * weight) {
    if (!weight) {
        return false;
    }
    const bool force_permute = htp_force_permute_name_from_env(weight->name);
    if (!htp_runtime_permute_qweights_enabled() && !force_permute) {
        return true;
    }

    const bool is_q4_like = weight->type == GGML_TYPE_Q4_0 || weight->type == GGML_TYPE_IQ4_NL;
    const bool is_q8_0    = weight->type == GGML_TYPE_Q8_0;
    if (!is_q4_like && !is_q8_0) {
        return true;
    }
    if (htp_skip_qkv_permute_repack(weight)) {
        return true;
    }
    if (!weight->data) {
        return false;
    }
    if (!ggml_is_contiguous(weight)) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP qweight permute skip (non-contiguous): name=%s type=%s\n", weight->name,
                    ggml_type_name(weight->type));
        }
        return false;
    }

    const size_t k     = (size_t) weight->ne[0];
    const size_t nrows = (size_t) ggml_nrows(weight);
    if (k == 0 || nrows == 0 || (k % 32) != 0 || (nrows % 32) != 0) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP qweight permute skip (shape): name=%s k=%zu n=%zu type=%s\n", weight->name, k, nrows,
                    ggml_type_name(weight->type));
        }
        return false;
    }

    static std::mutex                permute_mtx;
    static std::unordered_set<void*> permuted_weight_ptrs;
    const bool bypass_cache = htp_debug_disable_prepack_cache_enabled();

    std::lock_guard<std::mutex> guard(permute_mtx);
    if (!bypass_cache && permuted_weight_ptrs.count(weight->data) != 0) {
        return true;
    }

    auto * qtype = ggml_get_type_traits(weight->type);
    if (!qtype || !qtype->to_float) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    const size_t ne = k * nrows;
    std::vector<float> src_f32(ne);
    std::vector<float> permuted_f32(ne);
    qtype->to_float(weight->data, src_f32.data(), (int64_t) ne);
    htp_permute_linear_weight_f32(src_f32.data(), permuted_f32.data(), (int) nrows, (int) k);

    std::vector<float> imatrix(k, 1.0f);
    const size_t new_size = ggml_quantize_chunk(weight->type,
                                                permuted_f32.data(),
                                                weight->data,
                                                0,
                                                (int64_t) nrows,
                                                (int64_t) k,
                                                imatrix.data());
    const size_t expected_size = ggml_row_size(weight->type, k) * nrows;
    if (new_size != expected_size) {
        if (htp_debug_enabled()) {
            fprintf(stderr,
                    "HTP qweight permute failed (size mismatch): name=%s type=%s got=%zu expected=%zu\n",
                    weight->name, ggml_type_name(weight->type), new_size, expected_size);
        }
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    htp_prepack_stats_record(
        false,
        (uint64_t) ggml_nbytes(weight),
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    if (!bypass_cache) {
        permuted_weight_ptrs.insert(weight->data);
    }
    if (htp_debug_enabled()) {
        fprintf(stderr, "HTP qweight permuted in-place: name=%s type=%s k=%zu rows=%zu bypass_cache=%d\n", weight->name,
                ggml_type_name(weight->type), k, nrows, bypass_cache ? 1 : 0);
    }
    return true;
}

bool htp_repack_quant_weight_inplace_if_needed(struct ggml_tensor * weight) {
    if (!weight) {
        return false;
    }

    const bool is_q4_like = weight->type == GGML_TYPE_Q4_0 || weight->type == GGML_TYPE_IQ4_NL;
    const bool is_q8_0    = weight->type == GGML_TYPE_Q8_0;
    if (!is_q4_like && !is_q8_0) {
        return true;
    }
    if (htp_skip_qkv_permute_repack(weight)) {
        return true;
    }
    if (!htp_runtime_repack_qweights_enabled()) {
        return true;
    }
    if (!weight->data) {
        return false;
    }
    if (!ggml_is_contiguous(weight)) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP qweight repack skip (non-contiguous): name=%s type=%s\n", weight->name,
                    ggml_type_name(weight->type));
        }
        return false;
    }
    if (!htp_permute_quant_weight_inplace_if_needed(weight)) {
        return false;
    }

    const size_t k     = (size_t) weight->ne[0];
    const size_t nrows = (size_t) ggml_nrows(weight);
    constexpr size_t kSuperBlock = 256;
    if (k == 0 || k % kSuperBlock != 0) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP qweight repack skip (k %% 256 != 0): name=%s k=%zu type=%s\n", weight->name, k,
                    ggml_type_name(weight->type));
        }
        return false;
    }

    const size_t n_super_blocks = k / kSuperBlock;
    const size_t row_size       = ggml_row_size(weight->type, k);
    const size_t expected_row_size =
        is_q4_like ? n_super_blocks * sizeof(my_block_q4_0_like) : n_super_blocks * sizeof(my_block_q8_0);
    if (row_size != expected_row_size) {
        if (htp_debug_enabled()) {
            fprintf(stderr,
                    "HTP qweight repack skip (row-size mismatch): name=%s row_size=%zu expected=%zu type=%s\n",
                    weight->name, row_size, expected_row_size, ggml_type_name(weight->type));
        }
        return false;
    }

    static std::mutex                repack_mtx;
    static std::unordered_set<void*> repacked_weight_ptrs;
    const bool bypass_cache = htp_debug_disable_prepack_cache_enabled();

    std::lock_guard<std::mutex> guard(repack_mtx);
    if (!bypass_cache && repacked_weight_ptrs.count(weight->data) != 0) {
        return true;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto * base = reinterpret_cast<uint8_t *>(weight->data);
    for (size_t r = 0; r < nrows; ++r) {
        uint8_t * row = base + r * row_size;
        if (is_q4_like) {
            repack_row_q4_like_inplace(row, n_super_blocks);
        } else {
            repack_row_q8_0_inplace(row, n_super_blocks);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    htp_prepack_stats_record(
        true,
        (uint64_t) ggml_nbytes(weight),
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    if (!bypass_cache) {
        repacked_weight_ptrs.insert(weight->data);
    }
    if (htp_debug_enabled()) {
        fprintf(stderr, "HTP qweight repacked in-place: name=%s type=%s k=%zu rows=%zu bypass_cache=%d\n", weight->name,
                ggml_type_name(weight->type), k, nrows, bypass_cache ? 1 : 0);
    }
    return true;
}

bool htp_permute_f16_weight_inplace_if_needed(struct ggml_tensor * weight) {
    if (!weight) {
        return false;
    }
    if (weight->type != GGML_TYPE_F16) {
        return true;
    }
    if (!htp_runtime_permute_f16weights_enabled()) {
        return true;
    }
    if (!weight->data) {
        return false;
    }
    if (!ggml_is_contiguous(weight)) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP f16 weight permute skip (non-contiguous): name=%s\n", weight->name);
        }
        return false;
    }

    const size_t k     = (size_t) weight->ne[0];
    const size_t nrows = (size_t) ggml_nrows(weight);
    if (k == 0 || nrows == 0 || (k % 32) != 0 || (nrows % 32) != 0) {
        if (htp_debug_enabled()) {
            fprintf(stderr, "HTP f16 weight permute skip (shape): name=%s k=%zu n=%zu\n", weight->name, k, nrows);
        }
        return false;
    }

    static std::mutex                permute_mtx;
    static std::unordered_set<void*> permuted_weight_ptrs;
    const bool bypass_cache = htp_debug_disable_prepack_cache_enabled();
    std::lock_guard<std::mutex> guard(permute_mtx);
    if (!bypass_cache && permuted_weight_ptrs.count(weight->data) != 0) {
        return true;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const size_t ne = k * nrows;
    std::vector<float> src_f32(ne);
    std::vector<float> perm_f32(ne);
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(weight->data), src_f32.data(), (int64_t) ne);
    htp_permute_linear_weight_f32(src_f32.data(), perm_f32.data(), (int) nrows, (int) k);
    ggml_fp32_to_fp16_row(perm_f32.data(), reinterpret_cast<ggml_fp16_t *>(weight->data), (int64_t) ne);

    const auto t1 = std::chrono::steady_clock::now();
    htp_prepack_stats_record(
        false,
        (uint64_t) ggml_nbytes(weight),
        (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

    if (!bypass_cache) {
        permuted_weight_ptrs.insert(weight->data);
    }
    if (htp_debug_enabled()) {
        fprintf(stderr, "HTP f16 weight permuted in-place: name=%s k=%zu rows=%zu bypass_cache=%d\n",
                weight->name, k, nrows, bypass_cache ? 1 : 0);
    }
    return true;
}

bool htp_force_act_q8_for_weight_inplace_if_needed(const ggml_tensor * weight, struct ggml_tensor * activation) {
    if (!htp_qkv_force_act_q8_enabled()) {
        return true;
    }
    if (!weight || !activation) {
        return false;
    }
    const char * name = weight->name;
    if (name == nullptr || name[0] == '\0') {
        return true;
    }
    const bool is_qkv_family =
        std::strstr(name, ".attention.qkv.weight") != nullptr ||
        std::strstr(name, ".attention.q_proj.weight") != nullptr ||
        std::strstr(name, ".attention.k_proj.weight") != nullptr ||
        std::strstr(name, ".attention.v_proj.weight") != nullptr;
    if (!is_qkv_family) {
        return true;
    }
    if (activation->type != GGML_TYPE_F32 || !activation->data || !ggml_is_contiguous(activation)) {
        return false;
    }

    const int64_t k = activation->ne[0];
    const int64_t m = ggml_nrows(activation);
    if (k <= 0 || m <= 0) {
        return false;
    }

    const auto * q8_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const auto * q8_ref = ggml_get_type_traits(GGML_TYPE_Q8_0);
    ggml_from_float_t q8_from = q8_cpu ? q8_cpu->from_float : nullptr;
    if (htp_qkv_force_act_q8_ref_enabled() && q8_ref && q8_ref->from_float_ref) {
        q8_from = q8_ref->from_float_ref;
    }
    if (!q8_from || !q8_ref || !q8_ref->to_float) {
        return false;
    }

    const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, k);
    if (q8_row_size == 0) {
        return false;
    }

    std::vector<uint8_t> q8_buf((size_t) m * q8_row_size);
    float * act = reinterpret_cast<float *>(activation->data);
    for (int64_t row = 0; row < m; ++row) {
        float * act_row = act + row * k;
        void * q_row = q8_buf.data() + (size_t) row * q8_row_size;
        q8_from(act_row, q_row, k);
    }
    for (int64_t row = 0; row < m; ++row) {
        float * act_row = act + row * k;
        const void * q_row = q8_buf.data() + (size_t) row * q8_row_size;
        q8_ref->to_float(q_row, act_row, k);
    }
    if (htp_qkv_force_act_q8_copy_f16_enabled()) {
        const int64_t ne = m * k;
        std::vector<ggml_fp16_t> fp16_buf((size_t) ne);
        ggml_fp32_to_fp16_row(act, fp16_buf.data(), ne);
        ggml_fp16_to_fp32_row(fp16_buf.data(), act, ne);
    }

    if (htp_debug_enabled()) {
        static std::atomic<uint32_t> n_force_logs{ 0 };
        const uint32_t idx = n_force_logs.fetch_add(1, std::memory_order_relaxed) + 1;
        if (idx <= 8) {
            std::fprintf(stderr,
                         "HTP act-q8 force enabled: name=%s m=%lld k=%lld q8_ref=%d\n",
                         name, (long long) m, (long long) k,
                         htp_qkv_force_act_q8_ref_enabled() ? 1 : 0);
        }
    }

    return true;
}

struct HtpActivationOverrideBuf {
    void * ptr = nullptr;
    int fd = -1;
    size_t bytes = 0;
    bool mapped = false;
};

void htp_release_activation_override(HtpActivationOverrideBuf & ov) {
    if (ov.ptr == nullptr) {
        return;
    }
    if (ov.mapped && ov.fd >= 0 && ov.bytes > 0) {
        (void) fastrpc_munmap(CDSP_DOMAIN_ID, ov.fd, ov.ptr, ov.bytes);
    }
    rpcmem_free(ov.ptr);
    ov.ptr = nullptr;
    ov.fd = -1;
    ov.bytes = 0;
    ov.mapped = false;
}

bool htp_is_qkv_family_weight_name(const char * name) {
    if (!name || name[0] == '\0') {
        return false;
    }
    return std::strstr(name, ".attention.qkv.weight") != nullptr ||
           std::strstr(name, ".attention.q_proj.weight") != nullptr ||
           std::strstr(name, ".attention.k_proj.weight") != nullptr ||
           std::strstr(name, ".attention.v_proj.weight") != nullptr;
}

bool htp_prepare_force_act_q8_copy_for_weight(const ggml_tensor * weight, const ggml_tensor * activation,
                                               HtpActivationOverrideBuf & ov) {
    if (!htp_qkv_force_act_q8_copy_enabled()) {
        return false;
    }
    if (!weight || !activation) {
        return false;
    }
    const char * name = weight->name;
    if (!htp_is_qkv_family_weight_name(name)) {
        return false;
    }
    if (activation->type != GGML_TYPE_F32 || !activation->data || !ggml_is_contiguous(activation)) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-q8 copy force skipped (bad activation): name=%s\n", name);
        }
        return false;
    }

    const int64_t k = activation->ne[0];
    const int64_t m = ggml_nrows(activation);
    if (k <= 0 || m <= 0) {
        return false;
    }

    const auto * q8_cpu = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const auto * q8_ref = ggml_get_type_traits(GGML_TYPE_Q8_0);
    ggml_from_float_t q8_from = q8_cpu ? q8_cpu->from_float : nullptr;
    if (htp_qkv_force_act_q8_ref_enabled() && q8_ref && q8_ref->from_float_ref) {
        q8_from = q8_ref->from_float_ref;
    }
    if (!q8_from || !q8_ref || !q8_ref->to_float) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-q8 copy force skipped (missing q8 traits): name=%s\n", name);
        }
        return false;
    }

    const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, k);
    const size_t act_bytes = ggml_nbytes(activation);
    if (q8_row_size == 0 || act_bytes == 0) {
        return false;
    }

    void * tmp = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int) act_bytes);
    if (!tmp) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-q8 copy force alloc failed: name=%s bytes=%zu\n", name, act_bytes);
        }
        return false;
    }

    int tmp_fd = rpcmem_to_fd(tmp);
    if (tmp_fd < 0) {
        rpcmem_free(tmp);
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-q8 copy force to_fd failed: name=%s\n", name);
        }
        return false;
    }

    if (fastrpc_mmap(CDSP_DOMAIN_ID, tmp_fd, tmp, 0, act_bytes, FASTRPC_MAP_FD) != 0) {
        rpcmem_free(tmp);
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-q8 copy force mmap failed: name=%s\n", name);
        }
        return false;
    }

    std::memcpy(tmp, activation->data, act_bytes);
    float * act = reinterpret_cast<float *>(tmp);
    std::vector<uint8_t> q8_buf((size_t) m * q8_row_size);
    for (int64_t row = 0; row < m; ++row) {
        float * act_row = act + row * k;
        void * q_row = q8_buf.data() + (size_t) row * q8_row_size;
        q8_from(act_row, q_row, k);
    }
    for (int64_t row = 0; row < m; ++row) {
        float * act_row = act + row * k;
        const void * q_row = q8_buf.data() + (size_t) row * q8_row_size;
        q8_ref->to_float(q_row, act_row, k);
    }

    ov.ptr = tmp;
    ov.fd = tmp_fd;
    ov.bytes = act_bytes;
    ov.mapped = true;

    if (htp_debug_enabled()) {
        static std::atomic<uint32_t> n_force_logs{ 0 };
        const uint32_t idx = n_force_logs.fetch_add(1, std::memory_order_relaxed) + 1;
        if (idx <= 8) {
            std::fprintf(stderr,
                         "HTP act-q8 copy force enabled: name=%s m=%lld k=%lld bytes=%zu q8_ref=%d q8copy_fp16=%d\n",
                         name, (long long) m, (long long) k, act_bytes,
                         htp_qkv_force_act_q8_ref_enabled() ? 1 : 0,
                         htp_qkv_force_act_q8_copy_f16_enabled() ? 1 : 0);
        }
    }

    return true;
}

bool htp_prepare_force_act_f16_copy_for_weight(const ggml_tensor * weight, const ggml_tensor * activation,
                                               HtpActivationOverrideBuf & ov) {
    if (!htp_qkv_force_act_f16_copy_enabled()) {
        return false;
    }
    if (!weight || !activation) {
        return false;
    }
    const char * name = weight->name;
    if (!htp_is_qkv_family_weight_name(name)) {
        return false;
    }
    if (activation->type != GGML_TYPE_F32 || !activation->data || !ggml_is_contiguous(activation)) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-f16 copy force skipped (bad activation): name=%s\n", name);
        }
        return false;
    }

    const int64_t k = activation->ne[0];
    const int64_t m = ggml_nrows(activation);
    if (k <= 0 || m <= 0) {
        return false;
    }
    const int64_t ne = m * k;

    const size_t act_bytes = ggml_nbytes(activation);
    if (act_bytes == 0) {
        return false;
    }

    void * tmp = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int) act_bytes);
    if (!tmp) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-f16 copy force alloc failed: name=%s bytes=%zu\n", name, act_bytes);
        }
        return false;
    }

    int tmp_fd = rpcmem_to_fd(tmp);
    if (tmp_fd < 0) {
        rpcmem_free(tmp);
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-f16 copy force to_fd failed: name=%s\n", name);
        }
        return false;
    }

    if (fastrpc_mmap(CDSP_DOMAIN_ID, tmp_fd, tmp, 0, act_bytes, FASTRPC_MAP_FD) != 0) {
        rpcmem_free(tmp);
        if (htp_debug_enabled()) {
            std::fprintf(stderr, "HTP act-f16 copy force mmap failed: name=%s\n", name);
        }
        return false;
    }

    std::memcpy(tmp, activation->data, act_bytes);
    float * act = reinterpret_cast<float *>(tmp);
    std::vector<ggml_fp16_t> fp16_buf((size_t) ne);
    ggml_fp32_to_fp16_row(act, fp16_buf.data(), ne);
    ggml_fp16_to_fp32_row(fp16_buf.data(), act, ne);

    ov.ptr = tmp;
    ov.fd = tmp_fd;
    ov.bytes = act_bytes;
    ov.mapped = true;

    if (htp_debug_enabled()) {
        static std::atomic<uint32_t> n_force_logs{ 0 };
        const uint32_t idx = n_force_logs.fetch_add(1, std::memory_order_relaxed) + 1;
        if (idx <= 8) {
            std::fprintf(stderr,
                         "HTP act-f16 copy force enabled: name=%s m=%lld k=%lld bytes=%zu\n",
                         name, (long long) m, (long long) k, act_bytes);
        }
    }

    return true;
}

void htp_debug_log_supported_matmul(const ggml_tensor * dst, const ggml_tensor * weight, const ggml_tensor * activation) {
    static std::atomic<uint32_t> seen{ 0 };
    if (!htp_debug_enabled()) {
        return;
    }
    uint32_t idx = seen.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!htp_debug_log_all_matmul_enabled() && idx > 8) {
        return;
    }
    const char * buft_name = "(null)";
    if (weight->buffer && weight->buffer->buft && weight->buffer->buft->iface.get_name) {
        buft_name = weight->buffer->buft->iface.get_name(weight->buffer->buft);
    }
    fprintf(stderr,
            "supported matmul #%u: m=%d k=%zu n=%zu dst=%s w=%s a=%s weight_buft=%s w_name=%s alias(dst==a=%d,dst_overlap_a=%d)\n",
            idx,
            ggml_nrows(activation),
            (size_t) weight->ne[0],
            (size_t) weight->ne[1],
            ggml_type_name(dst->type),
            ggml_type_name(weight->type),
            ggml_type_name(activation->type),
            buft_name,
            (weight && weight->name[0] != '\0') ? weight->name : "<unnamed>",
            (dst && activation && dst->data == activation->data) ? 1 : 0,
            (dst && activation) ? (htp_ptr_ranges_overlap(dst->data, ggml_nbytes(dst), activation->data, ggml_nbytes(activation)) ? 1 : 0) : 0);
}

void htp_debug_log_matmul_skip(const char * reason, const ggml_tensor * dst, const ggml_tensor * weight, const ggml_tensor * activation) {
    if (!htp_debug_enabled()) {
        return;
    }
    static std::atomic<uint32_t> n_skip{ 0 };
    uint32_t idx = n_skip.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!(idx <= 128 || idx % 500 == 0)) {
        return;
    }
    const int m = activation ? ggml_nrows(activation) : -1;
    const int64_t k = weight ? weight->ne[0] : -1;
    const int64_t n = weight ? weight->ne[1] : -1;
    fprintf(stderr,
            "matmul skip #%u (%s): m=%d k=%lld n=%lld dst=%s w=%s a=%s w_name=%s\n",
            idx,
            reason ? reason : "unknown",
            m,
            (long long) k,
            (long long) n,
            dst ? ggml_type_name(dst->type) : "null",
            weight ? ggml_type_name(weight->type) : "null",
            activation ? ggml_type_name(activation->type) : "null",
            (weight && weight->name[0] != '\0') ? weight->name : "<unnamed>");
}

struct MatmulDebugSnapshot {
    bool done  = false;
    bool armed = false;
    uint64_t call_idx = 0;

    ggml_type weight_type = GGML_TYPE_COUNT;
    int op_index = -1;
    int m = 0;
    int k = 0;
    int n = 0;

    const ggml_tensor * activation = nullptr;
    const ggml_tensor * dst        = nullptr;
    std::string weight_name;

    std::vector<uint8_t> weight_raw;
};

MatmulDebugSnapshot & htp_matmul_debug_snapshot() {
    static MatmulDebugSnapshot snapshot;
    return snapshot;
}

static void htp_invert_permute_linear_weight_f32(const float * src_perm, float * dst_nonperm, int n, int k) {
    // Inverse of htp_permute_linear_weight_f32():
    // src_perm is [n,k] in the permuted layout; dst_nonperm is [n,k] in logical (ggml) layout.
    if (!src_perm || !dst_nonperm || (n % 32) != 0 || (k % 32) != 0) {
        return;
    }
    const int n_chunks = n / 32;
    const int k_chunks = k / 32;
    const int k_stride = k_chunks * 32;
    for (int r = 0; r < n; ++r) {
        const int64_t row_base = (int64_t) r * k_stride;
        for (int c = 0; c < k_stride; ++c) {
            int64_t t = row_base + c;
            int u     = (int) (t & 1);
            t >>= 1;
            int i = (int) (t & 31);
            t >>= 5;
            int g = (int) (t & 15);
            t >>= 4;
            int b = (int) (t % k_chunks);
            int a = (int) (t / k_chunks);
            if (a < 0 || a >= n_chunks) {
                continue;
            }
            int src_row = a * 32 + i;
            int src_col = b * 32 + g * 2 + u;
            dst_nonperm[(size_t) src_row * (size_t) k + (size_t) src_col] =
                src_perm[(size_t) r * (size_t) k + (size_t) c];
        }
    }
}

static bool htp_decode_packed_permuted_q4_like_weight_f32(ggml_type weight_type,
                                                          const uint8_t * raw,
                                                          size_t raw_size,
                                                          int n_rows,
                                                          int k_cols,
                                                          std::vector<float> & out_perm) {
    out_perm.clear();
    if (!raw || raw_size == 0 || n_rows <= 0 || k_cols <= 0 || (k_cols % 256) != 0) {
        return false;
    }
    const bool is_q4_like = (weight_type == GGML_TYPE_Q4_0) || (weight_type == GGML_TYPE_IQ4_NL);
    if (!is_q4_like) {
        return false;
    }

    const size_t n_super_blocks = (size_t) k_cols / 256;
    const size_t row_bytes      = n_super_blocks * sizeof(my_block_q4_0_like);
    const size_t expected       = row_bytes * (size_t) n_rows;
    if (raw_size != expected) {
        return false;
    }

    static const float iq4_nl_table[16] = {
        -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
        1.0f,    13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f,  113.0f,
    };
    const bool is_iq4_nl = weight_type == GGML_TYPE_IQ4_NL;

    out_perm.assign((size_t) n_rows * (size_t) k_cols, 0.0f);
    for (int row = 0; row < n_rows; ++row) {
        const uint8_t * row_ptr = raw + (size_t) row * row_bytes;
        float * dst_row = out_perm.data() + (size_t) row * (size_t) k_cols;
        for (size_t sb = 0; sb < n_super_blocks; ++sb) {
            const auto * blk = reinterpret_cast<const my_block_q4_0_like *>(row_ptr + sb * sizeof(my_block_q4_0_like));
            // Reconstruct the original 256 4-bit values (8 blocks x 32 vals) from the packed-quant layout.
            uint8_t qs[256];
            for (size_t j = 0; j < 64; ++j) {
                const uint8_t b0 = blk->quants[j * 2 + 0];
                const uint8_t b1 = blk->quants[j * 2 + 1];
                qs[j + 0]   = b0 & 0x0F;
                qs[j + 128] = b0 >> 4;
                qs[j + 64]  = b1 & 0x0F;
                qs[j + 192] = b1 >> 4;
            }
            for (size_t bi = 0; bi < 8; ++bi) {
                const float d = ggml_fp16_to_fp32((ggml_fp16_t) blk->scales[bi]);
                float * dst = dst_row + sb * 256 + bi * 32;
                for (size_t i = 0; i < 32; ++i) {
                    const uint8_t q = qs[bi * 32 + i];
                    const float v = is_iq4_nl ? iq4_nl_table[q] : (float) ((int) q - 8);
                    dst[i] = htp_round_to_fp16_scalar(v * d);
                }
            }
        }
    }
    return true;
}

static bool htp_decode_packed_permuted_q8_0_weight_f32(const uint8_t * raw,
                                                       size_t raw_size,
                                                       int n_rows,
                                                       int k_cols,
                                                       std::vector<float> & out_perm) {
    out_perm.clear();
    if (!raw || raw_size == 0 || n_rows <= 0 || k_cols <= 0 || (k_cols % 256) != 0) {
        return false;
    }
    const size_t n_super_blocks = (size_t) k_cols / 256;
    const size_t row_bytes      = n_super_blocks * sizeof(my_block_q8_0);
    const size_t expected       = row_bytes * (size_t) n_rows;
    if (raw_size != expected) {
        return false;
    }

    out_perm.assign((size_t) n_rows * (size_t) k_cols, 0.0f);
    for (int row = 0; row < n_rows; ++row) {
        const uint8_t * row_ptr = raw + (size_t) row * row_bytes;
        float * dst_row = out_perm.data() + (size_t) row * (size_t) k_cols;
        for (size_t sb = 0; sb < n_super_blocks; ++sb) {
            const auto * blk = reinterpret_cast<const my_block_q8_0 *>(row_ptr + sb * sizeof(my_block_q8_0));
            for (size_t bi = 0; bi < 8; ++bi) {
                const float d = ggml_fp16_to_fp32((ggml_fp16_t) blk->scales[bi]);
                const int8_t * qs = blk->quants + bi * 32;
                float * dst = dst_row + sb * 256 + bi * 32;
                for (size_t i = 0; i < 32; ++i) {
                    dst[i] = htp_round_to_fp16_scalar((float) qs[i] * d);
                }
            }
        }
    }
    return true;
}

void htp_debug_compare_matmul_snapshot(const MatmulDebugSnapshot & snap) {
    if (!snap.activation || !snap.dst || !snap.activation->data || !snap.dst->data || snap.weight_raw.empty()) {
        return;
    }

    const int m = snap.m;
    const int k = snap.k;
    const int n = snap.n;
    if (m <= 0 || k <= 0 || n <= 0) {
        return;
    }

    // MATMUL_DEBUG is O(m*n*k) and can take minutes on phone CPU for large DiT matmuls.
    // Allow evaluating only a top-left submatrix (still exact for that slice).
    int eval_m = m;
    int eval_n = n;
    const int env_eval_m = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_EVAL_M", 0);
    const int env_eval_n = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_EVAL_N", 0);
    if (env_eval_m > 0 && env_eval_m < eval_m) {
        eval_m = env_eval_m;
    }
    if (env_eval_n > 0 && env_eval_n < eval_n) {
        eval_n = env_eval_n;
    }

    // Decode weight in the *logical* (ggml) layout, regardless of HMX packed/permuted layout.
    std::vector<float> weight_f32((size_t) k * n, 0.0f);
    std::vector<float> weight_perm;
    const bool is_permuted_op =
        snap.op_index == HTP_OPS_MAT_MUL_PERMUTED_W16A32 ||
        snap.op_index == HTP_OPS_MAT_MUL_PERMUTED_W4D16A32 ||
        snap.op_index == HTP_OPS_MAT_MUL_PERMUTED_W8D16A32 ||
        snap.op_index == HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
    if (is_permuted_op) {
        // For PERMUTED matmul, weight_raw is expected to be in the packed/permuted layout.
        if (snap.weight_type == GGML_TYPE_F16) {
            weight_perm.assign((size_t) k * n, 0.0f);
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(snap.weight_raw.data()),
                                  weight_perm.data(), (int64_t) k * n);
        } else if (snap.weight_type == GGML_TYPE_Q4_0 || snap.weight_type == GGML_TYPE_IQ4_NL) {
            if (!htp_decode_packed_permuted_q4_like_weight_f32(snap.weight_type,
                                                              snap.weight_raw.data(),
                                                              snap.weight_raw.size(),
                                                              n,
                                                              k,
                                                              weight_perm)) {
                return;
            }
        } else if (snap.weight_type == GGML_TYPE_Q8_0) {
            if (!htp_decode_packed_permuted_q8_0_weight_f32(snap.weight_raw.data(),
                                                           snap.weight_raw.size(),
                                                           n,
                                                           k,
                                                           weight_perm)) {
                return;
            }
        } else {
            return;
        }

        // Recover logical (non-permuted) view.
        htp_invert_permute_linear_weight_f32(weight_perm.data(), weight_f32.data(), n, k);
    } else {
        // COMMON matmul: weight_raw is in the original ggml tensor layout.
        if (snap.weight_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(snap.weight_raw.data()), weight_f32.data(),
                                  (int64_t) k * n);
        } else {
            auto * qtype = ggml_get_type_traits(snap.weight_type);
            if (!qtype || !qtype->to_float) {
                return;
            }
            qtype->to_float(snap.weight_raw.data(), weight_f32.data(), (int64_t) k * n);
        }
        // For diagnostics parity, also compute the permuted view.
        if (k % 32 == 0 && n % 32 == 0) {
            weight_perm.assign((size_t) k * n, 0.0f);
            htp_permute_linear_weight_f32(weight_f32.data(), weight_perm.data(), n, k);
        }
    }

    // Diagnostics: inverse permute view (only meaningful when weight_perm is available).
    std::vector<float> weight_invperm;
    if (!weight_perm.empty() && (k % 32) == 0 && (n % 32) == 0) {
        weight_invperm.assign((size_t) k * n, 0.0f);
        htp_invert_permute_linear_weight_f32(weight_perm.data(), weight_invperm.data(), n, k);
    }

    std::vector<float> weight_perm_requant;
    if (!weight_perm.empty() &&
        (snap.weight_type == GGML_TYPE_Q4_0 || snap.weight_type == GGML_TYPE_Q8_0 || snap.weight_type == GGML_TYPE_IQ4_NL)) {
        auto * qtype = ggml_get_type_traits(snap.weight_type);
        if (qtype && qtype->to_float) {
            const size_t row_size = ggml_row_size(snap.weight_type, (int64_t) k);
            const size_t nbytes = row_size * (size_t) n;
            std::vector<uint8_t> qbuf(nbytes);
            std::vector<float> imatrix((size_t) k, 1.0f);
            const size_t qsz = ggml_quantize_chunk(snap.weight_type,
                                                   weight_perm.data(),
                                                   qbuf.data(),
                                                   0,
                                                   (int64_t) n,
                                                   (int64_t) k,
                                                   imatrix.data());
            if (qsz == nbytes) {
                weight_perm_requant.assign((size_t) k * n, 0.0f);
                qtype->to_float(qbuf.data(), weight_perm_requant.data(), (int64_t) k * n);
            }
        }
    }

    const auto * activation = reinterpret_cast<const float *>(snap.activation->data);
    const auto * output     = reinterpret_cast<const float *>(snap.dst->data);

    const int  raw_dump_call = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_RAW_DUMP_CALL", 0);
    const bool raw_dump_only = htp_env_int_cached("GGML_HTP_DEBUG_MATMUL_RAW_DUMP_ONLY", 0) != 0;
    if (raw_dump_call > 0 && (int) snap.call_idx == raw_dump_call) {
        const std::string base = std::string(htp_matmul_debug_dump_dir()) + "/htp_mmdebug_call_" +
                                 std::to_string(snap.call_idx);
        const bool act_raw_ok = htp_dump_raw_f32_to_file(base + "_act_raw.tensor", "activation_raw", activation, k, m);
        const bool out_raw_ok = htp_dump_raw_f32_to_file(base + "_out_raw.tensor", "output_raw", output, n, m);
        const bool weight_raw_ok =
            htp_dump_raw_bytes_tensor_to_file(base + "_weight_raw.tensor",
                                              snap.weight_name.empty() ? "weight_raw" : snap.weight_name.c_str(),
                                              snap.weight_type,
                                              snap.weight_raw.data(),
                                              snap.weight_raw.size(),
                                              k,
                                              n);

        FILE * meta = std::fopen((base + "_raw_meta.txt").c_str(), "wb");
        if (meta != nullptr) {
            std::fprintf(meta, "call=%llu\n",
                         (unsigned long long) snap.call_idx);
            std::fprintf(meta, "weight_name=%s\n",
                         snap.weight_name.empty() ? "<unnamed>" : snap.weight_name.c_str());
            std::fprintf(meta, "wtype=%s m=%d k=%d n=%d\n",
                         ggml_type_name(snap.weight_type), m, k, n);
            std::fprintf(meta, "dump_ok_act_raw=%d dump_ok_out_raw=%d dump_ok_weight_raw=%d\n",
                         act_raw_ok ? 1 : 0, out_raw_ok ? 1 : 0, weight_raw_ok ? 1 : 0);
            std::fclose(meta);
        }

        if (raw_dump_only) {
            std::fprintf(stderr,
                         "HTP_MATMUL_DEBUG raw_dump_only call=%llu wtype=%s m=%d k=%d n=%d w_name=%s "
                         "dump_ok_act_raw=%d dump_ok_out_raw=%d dump_ok_weight_raw=%d\n",
                         (unsigned long long) snap.call_idx,
                         ggml_type_name(snap.weight_type),
                         m,
                         k,
                         n,
                         snap.weight_name.empty() ? "<unnamed>" : snap.weight_name.c_str(),
                         act_raw_ok ? 1 : 0,
                         out_raw_ok ? 1 : 0,
                         weight_raw_ok ? 1 : 0);
            return;
        }
    }

    std::vector<float> activation_f16((size_t) m * k, 0.0f);
    {
        std::vector<ggml_fp16_t> tmp((size_t) m * k);
        ggml_fp32_to_fp16_row(activation, tmp.data(), (int64_t) m * k);
        ggml_fp16_to_fp32_row(tmp.data(), activation_f16.data(), (int64_t) m * k);
    }

    std::vector<float> weight_direct_f16((size_t) k * n, 0.0f);
    {
        std::vector<ggml_fp16_t> tmp((size_t) k * n);
        ggml_fp32_to_fp16_row(weight_f32.data(), tmp.data(), (int64_t) k * n);
        ggml_fp16_to_fp32_row(tmp.data(), weight_direct_f16.data(), (int64_t) k * n);
    }

    // Host-side bit-exact decode replica of DSP common_deq path (q4/q8),
    // used to separate "decode/layout mismatch" from arithmetic floor.
    std::vector<float> weight_common_deq_bitexact;
    const bool have_common_deq_bitexact =
        !is_permuted_op &&
        htp_decode_common_qk0_weight_bitexact(
            snap.weight_type,
            snap.weight_raw.data(),
            snap.weight_raw.size(),
            n,
            k,
            weight_common_deq_bitexact);

    auto round_to_fp16 = [](float v) -> float {
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
    };

    auto build_ref = [&](const std::vector<float> & weight_matrix, const float * act_matrix,
                         std::vector<float> * ref_f16acc32) {
        std::vector<float> ref((size_t) eval_m * eval_n, 0.0f);
        if (ref_f16acc32 != nullptr) {
            ref_f16acc32->assign((size_t) eval_m * eval_n, 0.0f);
        }
        for (int row = 0; row < eval_m; ++row) {
            const float * act_row = act_matrix + (size_t) row * k;
            float * ref_row       = ref.data() + (size_t) row * eval_n;
            for (int col = 0; col < eval_n; ++col) {
                const float * w_col = weight_matrix.data() + (size_t) col * k;
                double sum = 0.0;
                float sum_f16acc32 = 0.0f;
                float chunk_sum = 0.0f;
                for (int p = 0; p < k; ++p) {
                    const float prod = w_col[p] * act_row[p];
                    sum += (double) prod;
                    if (ref_f16acc32 != nullptr) {
                        chunk_sum += prod;
                        if (((p + 1) % 32) == 0 || (p + 1) == k) {
                            sum_f16acc32 = round_to_fp16(sum_f16acc32 + chunk_sum);
                            chunk_sum = 0.0f;
                        }
                    }
                }
                ref_row[col] = (float) sum;
                if (ref_f16acc32 != nullptr) {
                    (*ref_f16acc32)[(size_t) row * eval_n + col] = sum_f16acc32;
                }
            }
        }
        return ref;
    };

    std::vector<float> ref_direct_f16acc32;
    std::vector<float> ref_direct       = build_ref(weight_f32, activation, nullptr);
    std::vector<float> ref_direct_f16   = build_ref(weight_direct_f16, activation_f16.data(), &ref_direct_f16acc32);
    std::vector<float> ref_wf16_af32    = build_ref(weight_direct_f16, activation, nullptr);
    std::vector<float> ref_wf32_af16    = build_ref(weight_f32, activation_f16.data(), nullptr);
    std::vector<float> ref_common_deq_bitexact_f16;
    std::vector<float> ref_common_deq_bitexact_af32;
    std::vector<float> ref_direct_f16io = ref_direct_f16;
    {
        std::vector<ggml_fp16_t> tmp((size_t) eval_m * eval_n);
        ggml_fp32_to_fp16_row(ref_direct_f16io.data(), tmp.data(), (int64_t) eval_m * eval_n);
        ggml_fp16_to_fp32_row(tmp.data(), ref_direct_f16io.data(), (int64_t) eval_m * eval_n);
    }
    if (have_common_deq_bitexact) {
        ref_common_deq_bitexact_f16 = build_ref(weight_common_deq_bitexact, activation_f16.data(), nullptr);
        ref_common_deq_bitexact_af32 = build_ref(weight_common_deq_bitexact, activation, nullptr);
        std::vector<ggml_fp16_t> tmp((size_t) eval_m * eval_n);
        ggml_fp32_to_fp16_row(ref_common_deq_bitexact_f16.data(), tmp.data(), (int64_t) eval_m * eval_n);
        ggml_fp16_to_fp32_row(tmp.data(), ref_common_deq_bitexact_f16.data(), (int64_t) eval_m * eval_n);
    }
    // CPU-like reference: emulate ggml CPU vec-dot contract for this weight type,
    // including activation quantization to vec_dot_type (e.g. q4_0 x q8_0).
    std::vector<float> ref_cpu_vecdot;
    {
        const auto * t_w_cpu = ggml_get_type_traits_cpu(snap.weight_type);
        if (t_w_cpu && t_w_cpu->vec_dot != nullptr) {
            const enum ggml_type act_q_type = t_w_cpu->vec_dot_type;
            const auto * t_a_cpu = ggml_get_type_traits_cpu(act_q_type);
            if (t_a_cpu && t_a_cpu->from_float != nullptr) {
                const size_t w_row_size = ggml_row_size(snap.weight_type, (int64_t) k);
                const size_t a_row_size = ggml_row_size(act_q_type, (int64_t) k);
                if (w_row_size > 0 && a_row_size > 0 &&
                    snap.weight_raw.size() == (size_t) n * w_row_size) {
                    std::vector<uint8_t> act_q((size_t) eval_m * a_row_size);
                    for (int row = 0; row < eval_m; ++row) {
                        const float * act_row = activation + (size_t) row * k;
                        void * q_row = act_q.data() + (size_t) row * a_row_size;
                        t_a_cpu->from_float(act_row, q_row, (int64_t) k);
                    }

                    ref_cpu_vecdot.assign((size_t) eval_m * eval_n, 0.0f);
                    for (int row = 0; row < eval_m; ++row) {
                        const void * q_row = act_q.data() + (size_t) row * a_row_size;
                        for (int col = 0; col < eval_n; ++col) {
                            const void * w_row = snap.weight_raw.data() + (size_t) col * w_row_size;
                            float sum = 0.0f;
                            t_w_cpu->vec_dot(k, &sum, sizeof(float), w_row, 0, q_row, 0, 1);
                            ref_cpu_vecdot[(size_t) row * eval_n + col] = sum;
                        }
                    }
                }
            }
        }
    }
    std::vector<float> ref_invp         = weight_invperm.empty() ? std::vector<float>() : build_ref(weight_invperm, activation, nullptr);
    std::vector<float> ref_perm_layout  = weight_perm.empty() ? std::vector<float>() : build_ref(weight_perm, activation, nullptr);
    std::vector<float> ref_perm_requant = weight_perm_requant.empty() ? std::vector<float>() : build_ref(weight_perm_requant, activation, nullptr);

    auto eval_metrics = [&](const std::vector<float> & ref, bool transposed_view) {
        double mae = 0.0;
        double mse = 0.0;
        double max_abs = 0.0;
        double dot = 0.0;
        double norm_ref = 0.0;
        double norm_out = 0.0;
        double max_abs_ref = 0.0;
        double max_abs_out = 0.0;
        bool has_non_finite = false;

        for (int row = 0; row < eval_m; ++row) {
            for (int col = 0; col < eval_n; ++col) {
                const double r = (double) ref[(size_t) row * eval_n + col];
                const size_t out_idx = transposed_view ? ((size_t) col * m + row) : ((size_t) row * n + col);
                const double o = (double) output[out_idx];
                const double abs_r = std::fabs(r);
                const double abs_o = std::fabs(o);
                if (abs_r > max_abs_ref) {
                    max_abs_ref = abs_r;
                }
                if (abs_o > max_abs_out) {
                    max_abs_out = abs_o;
                }
                if (!std::isfinite(o)) {
                    has_non_finite = true;
                }
                const double d = std::fabs(r - o);
                mae += d;
                mse += d * d;
                if (d > max_abs) {
                    max_abs = d;
                }
                dot += r * o;
                norm_ref += r * r;
                norm_out += o * o;
            }
        }

        const double ne = (double) eval_m * eval_n;
        mae /= ne;
        mse /= ne;
        const double rmse = std::sqrt(mse);
        const double cos = dot / (std::sqrt(norm_ref) * std::sqrt(norm_out) + 1e-30);
        return std::tuple<double, double, double, double, int, double, double>(
            mae, rmse, max_abs, cos, has_non_finite ? 1 : 0, max_abs_ref, max_abs_out);
    };

    auto eval_pair_metrics = [&](const std::vector<float> & lhs, const std::vector<float> & rhs) {
        if (lhs.empty() || rhs.empty() || lhs.size() != rhs.size()) {
            return std::tuple<double, double, double, double, int>(0.0, 0.0, 0.0, 0.0, 1);
        }
        double mae = 0.0;
        double mse = 0.0;
        double max_abs = 0.0;
        double dot = 0.0;
        double norm_lhs = 0.0;
        double norm_rhs = 0.0;
        bool has_non_finite = false;
        const size_t ne = lhs.size();
        for (size_t i = 0; i < ne; ++i) {
            const double a = (double) lhs[i];
            const double b = (double) rhs[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                has_non_finite = true;
            }
            const double d = std::fabs(a - b);
            mae += d;
            mse += d * d;
            if (d > max_abs) {
                max_abs = d;
            }
            dot += a * b;
            norm_lhs += a * a;
            norm_rhs += b * b;
        }
        mae /= (double) ne;
        mse /= (double) ne;
        const double rmse = std::sqrt(mse);
        const double cos = dot / (std::sqrt(norm_lhs) * std::sqrt(norm_rhs) + 1e-30);
        return std::tuple<double, double, double, double, int>(mae, rmse, max_abs, cos, has_non_finite ? 1 : 0);
    };

    auto [mae, rmse, max_abs, cos, non_finite, max_ref, max_out] = eval_metrics(ref_direct, false);
    auto [mae_f16, rmse_f16, max_abs_f16, cos_f16, non_finite_f16, max_ref_f16, max_out_f16] = eval_metrics(ref_direct_f16, false);
    auto [mae_f16acc32, rmse_f16acc32, max_abs_f16acc32, cos_f16acc32, non_finite_f16acc32, max_ref_f16acc32, max_out_f16acc32] =
      eval_metrics(ref_direct_f16acc32, false);
    auto [mae_f16io, rmse_f16io, max_abs_f16io, cos_f16io, non_finite_f16io, max_ref_f16io, max_out_f16io] =
      eval_metrics(ref_direct_f16io, false);
    auto [mae_wf16_af32, rmse_wf16_af32, max_abs_wf16_af32, cos_wf16_af32, non_finite_wf16_af32, max_ref_wf16_af32, max_out_wf16_af32] =
      eval_metrics(ref_wf16_af32, false);
    auto [mae_wf32_af16, rmse_wf32_af16, max_abs_wf32_af16, cos_wf32_af16, non_finite_wf32_af16, max_ref_wf32_af16, max_out_wf32_af16] =
      eval_metrics(ref_wf32_af16, false);
    double mae_common_deq_f16 = 0.0, rmse_common_deq_f16 = 0.0, max_abs_common_deq_f16 = 0.0, cos_common_deq_f16 = 0.0;
    int non_finite_common_deq_f16 = 0;
    double max_ref_common_deq_f16 = 0.0, max_out_common_deq_f16 = 0.0;
    if (!ref_common_deq_bitexact_f16.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_common_deq_bitexact_f16, false);
        mae_common_deq_f16 = a;
        rmse_common_deq_f16 = b;
        max_abs_common_deq_f16 = c;
        cos_common_deq_f16 = d;
        non_finite_common_deq_f16 = e;
        max_ref_common_deq_f16 = f;
        max_out_common_deq_f16 = g;
    }
    double mae_common_deq_af32 = 0.0, rmse_common_deq_af32 = 0.0, max_abs_common_deq_af32 = 0.0, cos_common_deq_af32 = 0.0;
    int non_finite_common_deq_af32 = 0;
    double max_ref_common_deq_af32 = 0.0, max_out_common_deq_af32 = 0.0;
    if (!ref_common_deq_bitexact_af32.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_common_deq_bitexact_af32, false);
        mae_common_deq_af32 = a;
        rmse_common_deq_af32 = b;
        max_abs_common_deq_af32 = c;
        cos_common_deq_af32 = d;
        non_finite_common_deq_af32 = e;
        max_ref_common_deq_af32 = f;
        max_out_common_deq_af32 = g;
    }
    double mae_cdeq_f16io_vs_directf16io = 0.0, rmse_cdeq_f16io_vs_directf16io = 0.0, max_cdeq_f16io_vs_directf16io = 0.0, cos_cdeq_f16io_vs_directf16io = 0.0;
    int nf_cdeq_f16io_vs_directf16io = 0;
    if (!ref_common_deq_bitexact_f16.empty()) {
        auto [a, b, c, d, e] = eval_pair_metrics(ref_common_deq_bitexact_f16, ref_direct_f16io);
        mae_cdeq_f16io_vs_directf16io = a;
        rmse_cdeq_f16io_vs_directf16io = b;
        max_cdeq_f16io_vs_directf16io = c;
        cos_cdeq_f16io_vs_directf16io = d;
        nf_cdeq_f16io_vs_directf16io = e;
    }
    double mae_cdeq_af32_vs_wf16af32 = 0.0, rmse_cdeq_af32_vs_wf16af32 = 0.0, max_cdeq_af32_vs_wf16af32 = 0.0, cos_cdeq_af32_vs_wf16af32 = 0.0;
    int nf_cdeq_af32_vs_wf16af32 = 0;
    if (!ref_common_deq_bitexact_af32.empty()) {
        auto [a, b, c, d, e] = eval_pair_metrics(ref_common_deq_bitexact_af32, ref_wf16_af32);
        mae_cdeq_af32_vs_wf16af32 = a;
        rmse_cdeq_af32_vs_wf16af32 = b;
        max_cdeq_af32_vs_wf16af32 = c;
        cos_cdeq_af32_vs_wf16af32 = d;
        nf_cdeq_af32_vs_wf16af32 = e;
    }
    auto [mae_t, rmse_t, max_abs_t, cos_t, non_finite_t, max_ref_t, max_out_t] = eval_metrics(ref_direct, true);
    double mae_cpu_vecdot = 0.0, rmse_cpu_vecdot = 0.0, max_abs_cpu_vecdot = 0.0, cos_cpu_vecdot = 0.0;
    int non_finite_cpu_vecdot = 0;
    double max_ref_cpu_vecdot = 0.0, max_out_cpu_vecdot = 0.0;
    if (!ref_cpu_vecdot.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_cpu_vecdot, false);
        mae_cpu_vecdot = a;
        rmse_cpu_vecdot = b;
        max_abs_cpu_vecdot = c;
        cos_cpu_vecdot = d;
        non_finite_cpu_vecdot = e;
        max_ref_cpu_vecdot = f;
        max_out_cpu_vecdot = g;
    }

    double mae_inv = 0.0, rmse_inv = 0.0, max_abs_inv = 0.0, cos_inv = 0.0;
    int non_finite_inv = 0;
    if (!ref_invp.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_invp, false);
        mae_inv = a;
        rmse_inv = b;
        max_abs_inv = c;
        cos_inv = d;
        non_finite_inv = e;
    }

    double mae_perm = 0.0, rmse_perm = 0.0, max_abs_perm = 0.0, cos_perm = 0.0;
    int non_finite_perm = 0;
    if (!ref_perm_layout.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_perm_layout, false);
        mae_perm = a;
        rmse_perm = b;
        max_abs_perm = c;
        cos_perm = d;
        non_finite_perm = e;
    }

    double mae_perm_q = 0.0, rmse_perm_q = 0.0, max_abs_perm_q = 0.0, cos_perm_q = 0.0;
    int non_finite_perm_q = 0;
    if (!ref_perm_requant.empty()) {
        auto [a, b, c, d, e, f, g] = eval_metrics(ref_perm_requant, false);
        mae_perm_q = a;
        rmse_perm_q = b;
        max_abs_perm_q = c;
        cos_perm_q = d;
        non_finite_perm_q = e;
    }

    {
        const int dump_call = htp_matmul_debug_dump_call();
        if (dump_call > 0 && (int) snap.call_idx == dump_call) {
            if (eval_m != m || eval_n != n) {
                std::fprintf(stderr,
                             "HTP_MATMUL_DEBUG dump_call=%d skipped: eval_m/n (%d,%d) != full (%d,%d)\n",
                             dump_call, eval_m, eval_n, m, n);
                return;
            }
            const std::string base = std::string(htp_matmul_debug_dump_dir()) + "/htp_mmdebug_call_" +
                                     std::to_string(snap.call_idx);
            const bool act_ok = htp_dump_raw_f32_to_file(base + "_act.tensor", "activation", activation, k, m);
            const bool out_ok = htp_dump_raw_f32_to_file(base + "_out.tensor", "output", output, n, m);
            const bool ref_ok = htp_dump_raw_f32_to_file(base + "_ref_direct.tensor", "ref_direct",
                                                         ref_direct.data(), n, m);
            const bool ref_f16io_ok = htp_dump_raw_f32_to_file(base + "_ref_direct_f16io.tensor",
                                                                "ref_direct_f16io",
                                                                ref_direct_f16io.data(), n, m);
            const bool ref_wf16_af32_ok = htp_dump_raw_f32_to_file(base + "_ref_wf16_af32.tensor",
                                                                    "ref_wf16_af32",
                                                                    ref_wf16_af32.data(), n, m);
            bool ref_common_af32_ok = false;
            if (!ref_common_deq_bitexact_af32.empty()) {
                ref_common_af32_ok = htp_dump_raw_f32_to_file(base + "_ref_common_af32.tensor",
                                                              "ref_common_af32",
                                                              ref_common_deq_bitexact_af32.data(), n, m);
            }
            bool ref_cpu_vecdot_ok = false;
            if (!ref_cpu_vecdot.empty()) {
                ref_cpu_vecdot_ok = htp_dump_raw_f32_to_file(base + "_ref_cpu_vecdot.tensor",
                                                             "ref_cpu_vecdot",
                                                             ref_cpu_vecdot.data(), n, m);
            }

            FILE * meta = std::fopen((base + "_meta.txt").c_str(), "wb");
            if (meta != nullptr) {
                std::fprintf(meta, "call=%llu\n",
                             (unsigned long long) snap.call_idx);
                std::fprintf(meta, "weight_name=%s\n",
                             snap.weight_name.empty() ? "<unnamed>" : snap.weight_name.c_str());
                std::fprintf(meta, "wtype=%s m=%d k=%d n=%d\n",
                             ggml_type_name(snap.weight_type), m, k, n);
                std::fprintf(meta, "dump_ok_act=%d dump_ok_out=%d dump_ok_ref=%d dump_ok_ref_f16io=%d "
                                   "dump_ok_ref_wf16af32=%d dump_ok_ref_commonaf32=%d dump_ok_ref_cpuvecdot=%d\n",
                             act_ok ? 1 : 0, out_ok ? 1 : 0, ref_ok ? 1 : 0, ref_f16io_ok ? 1 : 0,
                             ref_wf16_af32_ok ? 1 : 0, ref_common_af32_ok ? 1 : 0, ref_cpu_vecdot_ok ? 1 : 0);
                std::fprintf(meta, "mae_direct=%.9g mae_direct_f16io=%.9g mae_wf16_af32=%.9g mae_common_af32=%.9g "
                                   "mae_cpu_vecdot=%.9g\n",
                             mae, mae_f16io, mae_wf16_af32, mae_common_deq_af32, mae_cpu_vecdot);
                std::fclose(meta);
            }
        }
    }

    fprintf(stderr,
            "HTP_MATMUL_DEBUG call=%llu wtype=%s m=%d k=%d n=%d eval_m=%d eval_n=%d w_name=%s "
            "direct(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "direct_f16ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "direct_f16acc32_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "direct_f16io_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "transpose_view(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "invperm_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d) "
            "perm_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d) "
            "perm_requant_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d) "
            "wf16_af32_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "wf32_af16_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "common_deq_bitexact_f16io_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "common_deq_bitexact_af32_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g) "
            "ref_pair(common_f16io_vs_direct_f16io_mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d "
            "common_af32_vs_wf16af32_mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d)\n",
            (unsigned long long) snap.call_idx, ggml_type_name(snap.weight_type), m, k, n, eval_m, eval_n,
            snap.weight_name.empty() ? "<unnamed>" : snap.weight_name.c_str(),
            mae, rmse, max_abs, cos, non_finite, max_ref, max_out,
            mae_f16, rmse_f16, max_abs_f16, cos_f16, non_finite_f16, max_ref_f16, max_out_f16,
            mae_f16acc32, rmse_f16acc32, max_abs_f16acc32, cos_f16acc32, non_finite_f16acc32, max_ref_f16acc32, max_out_f16acc32,
            mae_f16io, rmse_f16io, max_abs_f16io, cos_f16io, non_finite_f16io, max_ref_f16io, max_out_f16io,
            mae_t, rmse_t, max_abs_t, cos_t, non_finite_t, max_ref_t, max_out_t,
            mae_inv, rmse_inv, max_abs_inv, cos_inv, non_finite_inv,
            mae_perm, rmse_perm, max_abs_perm, cos_perm, non_finite_perm,
            mae_perm_q, rmse_perm_q, max_abs_perm_q, cos_perm_q, non_finite_perm_q,
            mae_wf16_af32, rmse_wf16_af32, max_abs_wf16_af32, cos_wf16_af32, non_finite_wf16_af32, max_ref_wf16_af32, max_out_wf16_af32,
            mae_wf32_af16, rmse_wf32_af16, max_abs_wf32_af16, cos_wf32_af16, non_finite_wf32_af16, max_ref_wf32_af16, max_out_wf32_af16,
            mae_common_deq_f16, rmse_common_deq_f16, max_abs_common_deq_f16, cos_common_deq_f16, non_finite_common_deq_f16, max_ref_common_deq_f16, max_out_common_deq_f16,
            mae_common_deq_af32, rmse_common_deq_af32, max_abs_common_deq_af32, cos_common_deq_af32, non_finite_common_deq_af32, max_ref_common_deq_af32, max_out_common_deq_af32,
            mae_cdeq_f16io_vs_directf16io, rmse_cdeq_f16io_vs_directf16io, max_cdeq_f16io_vs_directf16io, cos_cdeq_f16io_vs_directf16io, nf_cdeq_f16io_vs_directf16io,
            mae_cdeq_af32_vs_wf16af32, rmse_cdeq_af32_vs_wf16af32, max_cdeq_af32_vs_wf16af32, cos_cdeq_af32_vs_wf16af32, nf_cdeq_af32_vs_wf16af32);

    if (!ref_cpu_vecdot.empty()) {
        fprintf(stderr,
                "HTP_MATMUL_DEBUG_CPU_LIKE call=%llu wtype=%s m=%d k=%d n=%d eval_m=%d eval_n=%d w_name=%s "
                "cpu_vecdot_ref(mae=%.8g rmse=%.8g max_abs=%.8g cos=%.8g non_finite=%d max_ref=%.8g max_out=%.8g)\n",
                (unsigned long long) snap.call_idx, ggml_type_name(snap.weight_type), m, k, n, eval_m, eval_n,
                snap.weight_name.empty() ? "<unnamed>" : snap.weight_name.c_str(),
                mae_cpu_vecdot, rmse_cpu_vecdot, max_abs_cpu_vecdot, cos_cpu_vecdot, non_finite_cpu_vecdot,
                max_ref_cpu_vecdot, max_out_cpu_vecdot);
    }
}

struct SyntheticFlashMask {
    void * ptr   = nullptr;
    int    fd    = -1;
    size_t bytes = 0;
};

// Cache synthetic zero masks for FLASH_ATTN when graph mask is null.
// Key: upper 32 bits = qo_len, lower 32 bits = kv_len.
SyntheticFlashMask get_or_create_synthetic_flash_mask(int qo_len, int kv_len) {
    static std::mutex                                       mtx;
    static std::unordered_map<uint64_t, SyntheticFlashMask> cache;

    uint64_t key = ((uint64_t) (uint32_t) qo_len << 32) | (uint32_t) kv_len;
    std::lock_guard<std::mutex> lock(mtx);

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    size_t bytes = (size_t) qo_len * (size_t) kv_len * sizeof(uint16_t);
    void * ptr   = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int) bytes);
    if (ptr == nullptr) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr,
                         "htp flash synthetic mask alloc failed: qo=%d kv=%d bytes=%zu\n",
                         qo_len, kv_len, bytes);
        }
        return {};
    }
    memset(ptr, 0, bytes);

    int fd = rpcmem_to_fd(ptr);
    if (fd < 0) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr,
                         "htp flash synthetic mask fd failed: qo=%d kv=%d bytes=%zu\n",
                         qo_len, kv_len, bytes);
        }
        rpcmem_free(ptr);
        return {};
    }

    int err = fastrpc_mmap(CDSP_DOMAIN_ID, fd, ptr, 0, bytes, FASTRPC_MAP_FD);
    if (err != 0) {
        if (htp_debug_enabled()) {
            std::fprintf(stderr,
                         "htp flash synthetic mask mmap failed: qo=%d kv=%d bytes=%zu err=%d\n",
                         qo_len, kv_len, bytes, err);
        }
        rpcmem_free(ptr);
        return {};
    }

    SyntheticFlashMask mask{
        .ptr   = ptr,
        .fd    = fd,
        .bytes = bytes,
    };
    cache.emplace(key, mask);
    return mask;
}

void htp_op_stats_record_op(enum ggml_op op) {
    auto & stats = htp_op_stats();
    if (!stats.enabled) {
        return;
    }
    stats.offload_total.fetch_add(1, std::memory_order_relaxed);
    switch (op) {
        case GGML_OP_MUL_MAT:
            stats.offload_mul_mat.fetch_add(1, std::memory_order_relaxed);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            stats.offload_flash_attn.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            stats.offload_other.fetch_add(1, std::memory_order_relaxed);
            break;
    }
}

void htp_op_stats_record_latency(uint64_t us) {
    auto & stats = htp_op_stats();
    if (!stats.enabled) {
        return;
    }
    stats.offload_us.fetch_add(us, std::memory_order_relaxed);
}

void htp_op_stats_dump() {
    auto & stats = htp_op_stats();
    if (!stats.enabled) {
        return;
    }
    uint64_t n_total = stats.offload_total.load(std::memory_order_relaxed);
    uint64_t n_mm    = stats.offload_mul_mat.load(std::memory_order_relaxed);
    uint64_t n_fa    = stats.offload_flash_attn.load(std::memory_order_relaxed);
    uint64_t n_other = stats.offload_other.load(std::memory_order_relaxed);
    uint64_t us      = stats.offload_us.load(std::memory_order_relaxed);
    uint64_t n_umm   = stats.unsupported_mul_mat.load(std::memory_order_relaxed);
    uint64_t n_fseen = stats.flash_seen.load(std::memory_order_relaxed);
    uint64_t n_fnull = stats.flash_mask_null.load(std::memory_order_relaxed);
    uint64_t n_ftype = stats.flash_type_mismatch.load(std::memory_order_relaxed);
    fprintf(stderr,
            "GGML_HTP_STATS: offloaded_ops=%llu mul_mat=%llu flash_attn=%llu other=%llu "
            "offload_time=%.3fs unsupported_mul_mat=%llu flash_seen=%llu flash_mask_null=%llu flash_type_mismatch=%llu\n",
            (unsigned long long) n_total,
            (unsigned long long) n_mm,
            (unsigned long long) n_fa,
            (unsigned long long) n_other,
            us / 1e6,
            (unsigned long long) n_umm,
            (unsigned long long) n_fseen,
            (unsigned long long) n_fnull,
            (unsigned long long) n_ftype);
}

void htp_prepack_stats_dump() {
    auto & stats = htp_prepack_stats();
    if (!stats.enabled) {
        return;
    }
    const uint64_t permute_calls = stats.permute_calls.load(std::memory_order_relaxed);
    const uint64_t repack_calls  = stats.repack_calls.load(std::memory_order_relaxed);
    const uint64_t permute_bytes = stats.permute_bytes.load(std::memory_order_relaxed);
    const uint64_t repack_bytes  = stats.repack_bytes.load(std::memory_order_relaxed);
    const uint64_t permute_us    = stats.permute_us.load(std::memory_order_relaxed);
    const uint64_t repack_us     = stats.repack_us.load(std::memory_order_relaxed);

    std::fprintf(stderr,
                 "GGML_HTP_PREPACK: permute_calls=%" PRIu64 " permute_bytes=%" PRIu64 " permute_time=%.3fs "
                 "repack_calls=%" PRIu64 " repack_bytes=%" PRIu64 " repack_time=%.3fs total_time=%.3fs\n",
                 permute_calls, permute_bytes, permute_us / 1e6,
                 repack_calls, repack_bytes, repack_us / 1e6,
                 (permute_us + repack_us) / 1e6);
}

void htp_fallback_stats_dump() {
    auto & stats = htp_fallback_stats();
    if (!stats.enabled) {
        return;
    }
    const char * contract_id = std::getenv("GGML_HTP_CONTRACT_ACTIVE");
    if (contract_id == nullptr || contract_id[0] == '\0') {
        contract_id = std::getenv("GGML_HTP_CONTRACT_EXPECT");
    }
    if (contract_id == nullptr || contract_id[0] == '\0') {
        contract_id = "<unset>";
    }

    uint64_t total = 0;
    for (size_t i = 0; i < HTP_FALLBACK_REASON_COUNT; ++i) {
        total += stats.reason_counts[i].load(std::memory_order_relaxed);
    }
    if (total == 0) {
        return;
    }

    std::fprintf(stderr, "GGML_HTP_FALLBACK: total=%llu contract_id=%s\n",
                 (unsigned long long) total, contract_id);
    for (size_t i = 0; i < HTP_FALLBACK_REASON_COUNT; ++i) {
        const uint64_t n = stats.reason_counts[i].load(std::memory_order_relaxed);
        if (n == 0) {
            continue;
        }
        std::fprintf(stderr, "GGML_HTP_FALLBACK: reason=%s count=%llu\n",
                     htp_fallback_reason_name(static_cast<HtpFallbackReason>(i)),
                     (unsigned long long) n);
    }

    if (stats.csv_path[0] == '\0') {
        return;
    }

    FILE * fp = std::fopen(stats.csv_path, "w");
    if (!fp) {
        std::fprintf(stderr, "GGML_HTP_FALLBACK: failed to open csv '%s'\n", stats.csv_path);
        return;
    }

    std::fprintf(fp, "contract_id,%s\n", contract_id);
    std::fprintf(fp, "reason,count\n");
    for (size_t i = 0; i < HTP_FALLBACK_REASON_COUNT; ++i) {
        const uint64_t n = stats.reason_counts[i].load(std::memory_order_relaxed);
        if (n == 0) {
            continue;
        }
        std::fprintf(fp, "%s,%" PRIu64 "\n",
                     htp_fallback_reason_name(static_cast<HtpFallbackReason>(i)), n);
    }
    std::fprintf(fp, "\n");
    std::fprintf(fp, "op,reason,count\n");
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        for (size_t i = 0; i < HTP_FALLBACK_REASON_COUNT; ++i) {
            const uint64_t n = stats.op_reason_counts[op][i].load(std::memory_order_relaxed);
            if (n == 0) {
                continue;
            }
            std::fprintf(fp, "%s,%s,%" PRIu64 "\n",
                         ggml_op_name((enum ggml_op) op),
                         htp_fallback_reason_name(static_cast<HtpFallbackReason>(i)),
                         n);
        }
    }
    std::fclose(fp);
    std::fprintf(stderr, "GGML_HTP_FALLBACK: wrote csv to %s\n", stats.csv_path);
}

auto get_all_rpcmem_mappings(const ggml_tensor * dst) {
    const auto & mapper = ggml_backend_htp_context::instance()->mapper;

    std::vector<std::pair<int, ssize_t>> mappings;
    if (ggml_backend_buft_is_rpcmem(dst->buffer->buft)) {
        mappings.push_back(mapper.get_tensor_mapping(dst));
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        auto * src = dst->src[i];
        if (src && ggml_backend_buft_is_rpcmem(src->buffer->buft)) {
            mappings.push_back(mapper.get_tensor_mapping(src));
        }
    }
    return mappings;
}

template <typename T> void write_buf(uint8_t *& p, const T & v) {
    *reinterpret_cast<T *>(p) = v;
    p += sizeof(v);
}

void write_buf(uint8_t *& p, void * src, size_t size) {
    std::memcpy((void *) p, src, size);
    p += size;
}

uint8_t param_buf[4096];  // TODO(hzx): better implementation

int htp_ops_issue_request(ggml_backend_htp_context * ctx, int op_index, const void * args, int args_size) {
    int  n_reqs         = 1;
    int  n_unmap_fds    = ctx->mapper.get_pending_unmap_reqs().size();
    bool has_unmap_reqs = n_unmap_fds > 0;
    if (has_unmap_reqs) {
        ++n_reqs;
    }

    size_t op_req_size = sizeof(RequestHeader) + sizeof(OpComputeRequest) + args_size;

    auto * msg_hdr = reinterpret_cast<MessageHeader *>(ctx->ops_msg_chan);

    // FIXME: this is very ugly
    auto * d_ptr = reinterpret_cast<volatile std::atomic<uint64_t> *>(&(msg_hdr->state.d));

    // The memory order here is not very important
    std::atomic_store(d_ptr, static_cast<uint64_t>(0));

    msg_hdr->n_reqs         = n_reqs;
    msg_hdr->req_offsets[0] = message_header_size(msg_hdr);
    msg_hdr->req_offsets[1] = msg_hdr->req_offsets[0] + op_req_size;

    {
        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_OP_COMPUTE,
        };
        OpComputeRequest op_req{
            .op = (uint32_t) op_index,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 0));
        write_buf(p, req_hdr);
        write_buf(p, op_req);
        write_buf(p, const_cast<void *>(args), (size_t) args_size);
    }

    if (has_unmap_reqs) {
        size_t map_req_size     = sizeof(RequestHeader) + sizeof(RpcmemMapRequest) + n_unmap_fds * sizeof(int32_t);
        msg_hdr->req_offsets[2] = msg_hdr->req_offsets[1] + map_req_size;

        RequestHeader req_hdr{
            .state = 0,
            .type  = REQUEST_TYPE_RPCMEM_MAP,
        };
        RpcmemMapRequest map_req{
            .n_puts = n_unmap_fds,
            .n_gets = 0,
        };

        auto * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg_hdr, 1));
        write_buf(p, req_hdr);
        write_buf(p, map_req);
        for (const auto & [fd, _base, _len] : ctx->mapper.get_pending_unmap_reqs()) {
            write_buf(p, fd);
        }
    }

    // compute checksum
    if (1) {
        uint32_t   sum   = 0;
        uint32_t * begin = ((uint32_t *) msg_hdr) + 3;  // skip state & checksum
        uint32_t * end   = ((uint32_t *) msg_hdr) + ggml_backend_htp_context::MAX_MSG_SIZE / 4;

        for (auto * p = begin; p < end; ++p) {
            sum += *p;
        }
        sum += 0x00000001 + 0x00000000;  // value of `state`

        msg_hdr->checksum = -sum;
    } else {
#ifdef __aarch64__
        asm volatile("dmb sy" ::: "memory");
#endif
    }

    // issue request
    auto * v0_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[0]));
    auto * v1_ptr = reinterpret_cast<volatile std::atomic<uint8_t> *>(&(msg_hdr->state.v[1]));

    // NOTE(hzx): make sure memory_order_release is used here to ensure all previous writes are valid
    std::atomic_store_explicit(v0_ptr, static_cast<uint8_t>(1), std::memory_order_release);

    // poll for response
    while (std::atomic_load_explicit(v1_ptr, std::memory_order_acquire) == 0) {
        // TODO(hzx): use cpu_relax here
        usleep(1);
    }
    d_ptr->store(0, std::memory_order_relaxed);

    if (has_unmap_reqs) {
        ctx->mapper.unmap_all_pending_buffers();
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    return message_header_get_request_ptr(msg_hdr, 0)->state;
}

}  // namespace

extern "C" {

bool htp_ops_support_op(const struct ggml_tensor * dst) {
    (void) htp_op_stats();
    auto * ctx = ggml_backend_htp_context::instance();
    static bool logged_skip = false;
    static bool logged_uninit = false;
    if (ctx->skip_htp_ops) {
        if (!logged_skip) {
            fprintf(stderr, "htp_ops_support_op: SKIP_HTP_OPS is enabled, all ops fallback to CPU\n");
            logged_skip = true;
        }
        htp_fallback_record(HtpFallbackReason::kSkipByEnv, dst);
        return false;
    }
    if (!ctx->ops_backend_initialized) {
        if (!logged_uninit) {
            fprintf(stderr, "htp_ops_support_op: ops backend not initialized, all ops fallback to CPU\n");
            logged_uninit = true;
        }
        htp_fallback_record(HtpFallbackReason::kBackendUninitialized, dst);
        return false;
    }

    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            htp_fallback_record(HtpFallbackReason::kRmsNormDisabled, dst);
            return false;

            if (dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
                // NOTE: RPC version is mainly for testing
                return dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32") != nullptr;
            }
            return false;
        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                size_t m = ggml_nrows(activation);
                constexpr size_t kVecAlign = 128;
                size_t k = weight->ne[0];
                size_t n = weight->ne[1];

                if (!htp_matmul_name_allowed(weight)) {
                    htp_debug_log_matmul_skip("name-filter", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulNameFilter, dst);
                    return false;
                }
                if (weight->name[0] != '\0' &&
                    std::strstr(weight->name, ".adaLN_modulation.") != nullptr &&
                    !htp_adaln_matmul_enabled()) {
                    htp_debug_log_matmul_skip("adaln-disabled", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulAdalnDisabled, dst);
                    return false;
                }
                if (weight->name[0] != '\0' &&
                    std::strstr(weight->name, ".cap_embedder.1.weight") != nullptr &&
                    !htp_cap_embed_matmul_enabled()) {
                    htp_debug_log_matmul_skip("cap-embed-disabled", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulNameFilter, dst);
                    return false;
                }
                if (weight->name[0] != '\0' &&
                    std::strstr(weight->name, ".noise_refiner.") != nullptr &&
                    std::strstr(weight->name, ".feed_forward.w2.weight") != nullptr &&
                    !htp_noise_refiner_w2_matmul_enabled()) {
                    htp_debug_log_matmul_skip("noise-refiner-w2-disabled", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulNameFilter, dst);
                    return false;
                }
                if ((int) m < htp_matmul_min_m() || (int) k < htp_matmul_min_k() || (int) n < htp_matmul_min_n()) {
                    htp_debug_log_matmul_skip("min-shape-filter", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulMinShape, dst);
                    return false;
                }

                bool shape_ok_common = n % 32 == 0 && ggml_nrows(dst) == dst->ne[1] && m == (size_t) activation->ne[1];
                bool contiguous_ok = ggml_is_contiguous(dst) && ggml_is_contiguous(weight) &&
                                     ggml_is_contiguous(activation);
                bool aligned_ok = htp_ptr_aligned(dst->data, kVecAlign) &&
                                  htp_ptr_aligned(weight->data, kVecAlign) &&
                                  htp_ptr_aligned(activation->data, kVecAlign);

                if (!contiguous_ok || !aligned_ok) {
                    if (htp_debug_enabled()) {
                        fprintf(stderr,
                                "matmul skip (%s%s): dst_c=%d w_c=%d a_c=%d dst_aligned=%d w_aligned=%d a_aligned=%d\n",
                                !contiguous_ok ? "non-contiguous" : "",
                                (!contiguous_ok && !aligned_ok) ? ", " : (!aligned_ok ? "unaligned" : ""),
                                ggml_is_contiguous(dst), ggml_is_contiguous(weight), ggml_is_contiguous(activation),
                                htp_ptr_aligned(dst->data, kVecAlign), htp_ptr_aligned(weight->data, kVecAlign),
                                htp_ptr_aligned(activation->data, kVecAlign));
                    }
                    htp_debug_log_matmul_skip("contiguous-or-align", dst, weight, activation);
                    htp_fallback_record(HtpFallbackReason::kMatmulContiguousOrAlign, dst);
                    return false;
                }

                // FP16 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (!htp_f16_matmul_enabled()) {
                        htp_debug_log_matmul_skip("f16-disabled", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulF16Disabled, dst);
                        return false;
                    }
                    bool shape_ok = shape_ok_common && k % 32 == 0;
                    if (shape_ok) {
                        htp_debug_log_supported_matmul(dst, weight, activation);
                    } else {
                        htp_debug_log_matmul_skip("f16-shape", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulF16Shape, dst);
                    }
                    return shape_ok;
                }
                // (repacked) Q4_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 && activation->type == GGML_TYPE_F32) {
                    if (!htp_quant_matmul_enabled()) {
                        htp_debug_log_matmul_skip("quant-disabled", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulQuantDisabled, dst);
                        return false;
                    }
                    bool shape_ok = shape_ok_common && k % 256 == 0;
                    if (shape_ok) {
                        htp_debug_log_supported_matmul(dst, weight, activation);
                    } else {
                        htp_debug_log_matmul_skip("q4-shape", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulQ4Shape, dst);
                    }
                    return shape_ok;
                }
                // (repacked) Q8_0 weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 && activation->type == GGML_TYPE_F32) {
                    if (!htp_quant_matmul_enabled()) {
                        htp_debug_log_matmul_skip("quant-disabled", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulQuantDisabled, dst);
                        return false;
                    }
                    bool shape_ok = shape_ok_common && k % 256 == 0;
                    if (shape_ok) {
                        htp_debug_log_supported_matmul(dst, weight, activation);
                    } else {
                        htp_debug_log_matmul_skip("q8-shape", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulQ8Shape, dst);
                    }
                    return shape_ok;
                }
                // (repacked) IQ4_NL weight
                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                    activation->type == GGML_TYPE_F32) {
                    if (!htp_quant_matmul_enabled()) {
                        htp_debug_log_matmul_skip("quant-disabled", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulQuantDisabled, dst);
                        return false;
                    }
                    bool shape_ok = shape_ok_common && k % 256 == 0;
                    if (shape_ok) {
                        htp_debug_log_supported_matmul(dst, weight, activation);
                    } else {
                        htp_debug_log_matmul_skip("iq4-shape", dst, weight, activation);
                        htp_fallback_record(HtpFallbackReason::kMatmulIQ4Shape, dst);
                    }
                    return shape_ok;
                }
                static std::atomic<uint32_t> unsupported_mm{ 0 };
                uint32_t n_unsupported = unsupported_mm.fetch_add(1, std::memory_order_relaxed) + 1;
                htp_op_stats().unsupported_mul_mat.fetch_add(1, std::memory_order_relaxed);
                if (htp_debug_enabled() && (n_unsupported <= 8 || n_unsupported % 1000 == 0)) {
                    int m = ggml_nrows(activation);
                    fprintf(stderr, "unsupported matmul #%u: m=%d k=%zu n=%zu dst: %s weight: %s act: %s\n", n_unsupported,
                            m, k, n,
                            ggml_type_name(dst->type), ggml_type_name(weight->type),
                            ggml_type_name(activation->type));
                }
                htp_fallback_record(HtpFallbackReason::kMatmulTypeUnsupported, dst);
                return false;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                if (!htp_flash_attn_enabled()) {
                    htp_fallback_record(HtpFallbackReason::kFlashDisabled, dst);
                    return false;
                }

                float scale         = *reinterpret_cast<const float *>(&dst->op_params[0]);
                float max_bias      = *reinterpret_cast<const float *>(&dst->op_params[1]);
                float logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);

                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * mask = dst->src[3];
                const uint32_t flash_flags = *reinterpret_cast<const uint32_t *>(&dst->op_params[4]);
                const bool flash_q_row_major = (flash_flags & GGML_HTP_FLASH_ATTN_FLAG_Q_ROW_MAJOR) != 0;
                const bool flash_k_row_major = (flash_flags & GGML_HTP_FLASH_ATTN_FLAG_K_ROW_MAJOR) != 0;

                // HTP flash-attn kernel does not receive tensor strides; it assumes a token-major physical
                // layout for Q/K/V. For ggml tensors with ne=[D, L, H, N], this corresponds to:
                //   - nb0 = elem
                //   - nb2 = D*elem        (head stride)
                //   - nb1 = H*nb2         (token stride)
                // This is typically a permuted view over a base contiguous [D, H, L, N] tensor.
                auto flash_io_layout_ok = [](const ggml_tensor * t) -> bool {
                    if (!t) {
                        return false;
                    }
                    const size_t elem = ggml_type_size(t->type);
                    const size_t D    = (size_t) t->ne[0];
                    const size_t L    = (size_t) t->ne[1];
                    const size_t H    = (size_t) t->ne[2];
                    if (D == 0 || L == 0 || H == 0) {
                        return false;
                    }
                    const size_t expect_nb0 = elem;
                    const size_t expect_nb2 = D * elem;
                    const size_t expect_nb1 = H * expect_nb2;
                    if ((size_t) t->nb[0] != expect_nb0) {
                        return false;
                    }
                    if ((size_t) t->nb[1] != expect_nb1) {
                        return false;
                    }
                    if ((size_t) t->nb[2] != expect_nb2) {
                        return false;
                    }
                    if (t->ne[3] > 1) {
                        const size_t expect_nb3 = expect_nb1 * L;
                        if ((size_t) t->nb[3] != expect_nb3) {
                            return false;
                        }
                    }
                    return true;
                };
                auto flash_qk_rowmajor_layout_ok = [](const ggml_tensor * t) -> bool {
                    if (!t) {
                        return false;
                    }
                    const size_t elem = ggml_type_size(t->type);
                    const size_t D    = (size_t) t->ne[0];
                    const size_t L    = (size_t) t->ne[1];
                    const size_t H    = (size_t) t->ne[2];
                    if (D == 0 || L == 0 || H == 0) {
                        return false;
                    }
                    if ((size_t) t->nb[0] != elem) {
                        return false;
                    }
                    if ((size_t) t->nb[1] != D * elem) {
                        return false;
                    }
                    if ((size_t) t->nb[2] != D * L * elem) {
                        return false;
                    }
                    if (t->ne[3] > 1) {
                        const size_t expect_nb3 = D * L * H * elem;
                        if ((size_t) t->nb[3] != expect_nb3) {
                            return false;
                        }
                    }
                    return true;
                };

                constexpr size_t kVecAlign = 128;
                bool contiguous_ok = ggml_is_contiguous(dst) &&
                                     (flash_q_row_major ? flash_qk_rowmajor_layout_ok(q) : flash_io_layout_ok(q)) &&
                                     (flash_k_row_major ? flash_qk_rowmajor_layout_ok(k) : flash_io_layout_ok(k)) &&
                                     flash_io_layout_ok(v) &&
                                     (mask == nullptr || ggml_is_contiguous(mask));
                bool aligned_ok = htp_ptr_aligned(dst->data, kVecAlign) && htp_ptr_aligned(q->data, kVecAlign) &&
                                  htp_ptr_aligned(k->data, kVecAlign) && htp_ptr_aligned(v->data, kVecAlign) &&
                                  (mask == nullptr || htp_ptr_aligned(mask->data, kVecAlign));
                if (!contiguous_ok || !aligned_ok) {
                    if (htp_debug_enabled()) {
                        static std::atomic<uint32_t> skip_logged{ 0 };
                        const uint32_t idx = skip_logged.fetch_add(1, std::memory_order_relaxed) + 1;
                        if (idx <= 4) {
                            auto dump_tensor = [&](const char * tag, const ggml_tensor * t) {
                                if (!t) {
                                    fprintf(stderr, "  %s: <null>\n", tag);
                                    return;
                                }
                                fprintf(stderr,
                                        "  %s: type=%s ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] data=%p layout_ok=%d aligned=%d\n",
                                        tag,
                                        ggml_type_name(t->type),
                                        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                                        (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3],
                                        t->data,
                                        flash_io_layout_ok(t) ? 1 : 0,
                                        htp_ptr_aligned(t->data, kVecAlign) ? 1 : 0);
                            };
                            fprintf(stderr,
                                    "flash skip #%u: contiguous_ok=%d aligned_ok=%d dst=%s\n",
                                    idx, contiguous_ok ? 1 : 0, aligned_ok ? 1 : 0, dst->name);
                            dump_tensor("dst", dst);
                            dump_tensor("q", q);
                            dump_tensor("k", k);
                            dump_tensor("v", v);
                            dump_tensor("mask", mask);
                        }
                    }
                    htp_fallback_record(HtpFallbackReason::kFlashContiguousOrAlign, dst);
                    return false;
                }

                htp_op_stats().flash_seen.fetch_add(1, std::memory_order_relaxed);

                bool ok = dst->type == GGML_TYPE_F32 && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                          v->type == GGML_TYPE_F16 && (mask == nullptr || mask->type == GGML_TYPE_F16) && max_bias == 0 &&
                          logit_softcap == 0;
                if (mask == nullptr) {
                    htp_op_stats().flash_mask_null.fetch_add(1, std::memory_order_relaxed);
                } else if (!ok) {
                    htp_op_stats().flash_type_mismatch.fetch_add(1, std::memory_order_relaxed);
                }
                if (!ok) {
                    htp_fallback_record(HtpFallbackReason::kFlashTypeContract, dst);
                }
                static std::atomic<uint32_t> flash_seen{ 0 };
                uint32_t n_flash = flash_seen.fetch_add(1, std::memory_order_relaxed) + 1;
                if (htp_debug_enabled() && n_flash <= 4) {
                    if (mask != nullptr) {
                        fprintf(stderr,
                                "flash_attn #%u support=%d q=[%ld,%ld,%ld] k=[%ld,%ld,%ld] v=[%ld,%ld,%ld] "
                                "mask=[%ld,%ld,%ld] scale=%g\n",
                                n_flash, ok ? 1 : 0,
                                q->ne[0], q->ne[1], q->ne[2],
                                k->ne[0], k->ne[1], k->ne[2],
                                v->ne[0], v->ne[1], v->ne[2],
                                mask->ne[0], mask->ne[1], mask->ne[2],
                                scale);
                    } else {
                        fprintf(stderr,
                                "flash_attn #%u support=%d q=[%ld,%ld,%ld] k=[%ld,%ld,%ld] v=[%ld,%ld,%ld] "
                                "mask=[null] scale=%g\n",
                                n_flash, ok ? 1 : 0,
                                q->ne[0], q->ne[1], q->ne[2],
                                k->ne[0], k->ne[1], k->ne[2],
                                v->ne[0], v->ne[1], v->ne[2],
                                scale);
                    }
                }
                return ok;
            }
        case GGML_OP_MAP_CUSTOM2:
            {
                if (!htp_is_zimg_rope_op(dst)) {
                    htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
                    return false;
                }
                if (!htp_zimg_rope_contract_ok(dst)) {
                    htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
                    return false;
                }
                return true;
            }
        case GGML_OP_MAP_CUSTOM3:
            {
                if (htp_is_zimg_qknorm_rope_op(dst)) {
                    if (!htp_zimg_qknorm_rope_contract_ok(dst)) {
                        htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
                        return false;
                    }
                    return true;
                }
                if (htp_is_flux_ss_linear2_fused_op(dst)) {
                    if (!htp_flux_ss_linear2_contract_ok(dst)) {
                        htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
                        return false;
                    }
                    return true;
                }
                htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
                return false;
            }
        default:
            htp_fallback_record(HtpFallbackReason::kUnknownOp, dst);
            return false;
    }
}

int htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (params->ith != 0) {
        return 0;
    }

    static std::mutex g_htp_msg_mutex;
    std::lock_guard<std::mutex> lock(g_htp_msg_mutex);

    prepare_tensor_rpcmem_mapping(dst);

    auto * ctx           = ggml_backend_htp_context::instance();
    void * ops_dl_handle = ctx->ops_dl_handle;
    GGML_ASSERT(ops_dl_handle);

    constexpr bool prefer_rpc = false;

    int op_index  = -1;
    int args_size = 0;  // strictly 32 bits
    bool matmul_params_ready = false;
    const ggml_tensor * matmul_weight = nullptr;
    MatMulParams matmul_params{};
    HtpActivationOverrideBuf act_override{};
    bool act_override_active = false;
    bool flash_dump_armed = false;
    int flash_dump_idx = -1;
    bool flash_hash_trace_armed = false;
    int flash_hash_trace_idx = -1;
    const ggml_tensor * flash_q = nullptr;
    const ggml_tensor * flash_k = nullptr;
    const ggml_tensor * flash_v = nullptr;
    const ggml_tensor * flash_mask = nullptr;
    float flash_scale = 0.0f;
    float flash_max_bias = 0.0f;
    float flash_logit_softcap = 0.0f;

    switch (dst->op) {
        case GGML_OP_RMS_NORM:
            {
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 2);

                auto [dst_fd, dst_offset] = mappings[0];
                auto [src_fd, src_offset] = mappings[1];

                if (prefer_rpc) {
                    using fn_type = int(int, int, int, int, int, int);

                    auto op_fn = reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_rms_norm_f32"));
                    GGML_ASSERT(op_fn);

                    return op_fn(dst_fd, dst_offset, src_fd, src_offset, dst->ne[0], ggml_nrows(dst));
                }

                RmsNormF32Params params{
                    .dst = { dst_fd, (int32_t) dst_offset },
                    .src = { src_fd, (int32_t) src_offset },
                    .ne0 = (int32_t) dst->ne[0],
                    .ne1 = (int32_t) ggml_nrows(dst),
                };
                *reinterpret_cast<RmsNormF32Params *>(param_buf) = params;

                op_index  = HTP_OPS_RMS_NORM_F32;
                args_size = sizeof(RmsNormF32Params);
            }
            break;

        case GGML_OP_MUL_MAT:
            {
                auto * weight     = dst->src[0];
                auto * activation = dst->src[1];

                auto & mm_debug = htp_matmul_debug_snapshot();
                static std::atomic<uint64_t> mm_debug_call_counter{ 0 };
                if (htp_matmul_debug_check_enabled() && (htp_matmul_debug_check_all_enabled() || !mm_debug.done) &&
                    (weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_Q4_0 ||
                     weight->type == GGML_TYPE_Q8_0 || weight->type == GGML_TYPE_IQ4_NL)) {
                    bool arm_this = true;
                    if (const char * filter = htp_matmul_debug_weight_contains_filter()) {
                        const char * w_name = (weight && weight->name[0] != '\0') ? weight->name : nullptr;
                        arm_this = (w_name && std::strstr(w_name, filter) != nullptr);
                    }
                    if (arm_this) {
                        mm_debug.armed      = true;
                        mm_debug.call_idx   = mm_debug_call_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                        mm_debug.weight_type = weight->type;
                        mm_debug.op_index   = -1;
                        mm_debug.activation = activation;
                        mm_debug.dst        = dst;
                        mm_debug.m          = ggml_nrows(activation);
                        mm_debug.k          = weight->ne[0];
                        mm_debug.n          = weight->ne[1];
                        mm_debug.weight_name = (weight->name[0] != '\0') ? weight->name : "";
                        // Defer weight snapshot until after any runtime permute/repack so that
                        // debug can validate offline-prepacked GGUFs as well.
                        mm_debug.weight_raw.clear();
                    }
                }

                GGML_ASSERT(htp_permute_f16_weight_inplace_if_needed(weight));
                GGML_ASSERT(htp_repack_quant_weight_inplace_if_needed(weight));
                if (!htp_qkv_force_act_q8_copy_enabled()) {
                    GGML_ASSERT(htp_force_act_q8_for_weight_inplace_if_needed(weight, activation));
                }

                htp_debug_dump_weight_if_requested(weight);

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [output_fd, output_offset]         = mappings[0];
                auto [weight_fd, weight_offset]         = mappings[1];
                auto [activation_fd, activation_offset] = mappings[2];

                int m = ggml_nrows(activation);
                int k = weight->ne[0];
                int n = weight->ne[1];

                act_override_active = htp_prepare_force_act_q8_copy_for_weight(weight, activation, act_override);
                if (!act_override_active) {
                    act_override_active = htp_prepare_force_act_f16_copy_for_weight(weight, activation, act_override);
                }
                if (act_override_active) {
                    activation_fd = act_override.fd;
                    activation_offset = 0;
                }

                MatMulParams mm_params{
                    .output     = { output_fd,     (int32_t) output_offset     },
                    .activation = { activation_fd, (int32_t) activation_offset },
                    .weight     = { weight_fd,     (int32_t) weight_offset     },
                    .m          = m,
                    .k          = k,
                    .n          = n,
                    .flags      = htp_q8_out_stationary_enabled() ? HTP_MATMUL_FLAG_Q8_OUT_STATIONARY : 0u,
                };
                if (weight->type == GGML_TYPE_Q8_0 &&
                    htp_q8_out_stationary_enabled() &&
                    mm_debug.armed &&
                    htp_q8_outstationary_firstblock_dump_selected(mm_debug.call_idx)) {
                    mm_params.flags |= HTP_MATMUL_FLAG_DBG_OUTSTAT_FIRSTBLOCK_DUMP;
                }
                if (weight->type == GGML_TYPE_Q8_0 &&
                    htp_q8_out_stationary_enabled() &&
                    mm_debug.armed &&
                    htp_q8_outstationary_acttile_dump_enabled() &&
                    htp_matmul_debug_raw_dump_only_selected(mm_debug.call_idx)) {
                    mm_params.flags |= HTP_MATMUL_FLAG_DBG_OUTSTAT_ACT_TILE_DUMP;
                    if (htp_q8_outstationary_act_direct_stage_enabled()) {
                        mm_params.flags |= HTP_MATMUL_FLAG_DBG_ACT_DIRECT_STAGE;
                    }
                    if (htp_q8_outstationary_act_scratch_direct_transfer_enabled()) {
                        mm_params.flags |= HTP_MATMUL_FLAG_DBG_ACT_SCRATCH_DIRECT_TRANSFER;
                    }
                }
                if (weight->type == GGML_TYPE_Q8_0 &&
                    htp_q8_out_stationary_enabled() &&
                    mm_debug.armed &&
                    htp_q8_outstationary_scratch_scalardump_enabled() &&
                    htp_matmul_debug_raw_dump_only_selected(mm_debug.call_idx)) {
                    mm_params.flags |= HTP_MATMUL_FLAG_DBG_OUTSTAT_SCRATCH_HVX_DUMP;
                    mm_params.flags |= HTP_MATMUL_FLAG_DBG_ACT_META_OUT;
                }
                *reinterpret_cast<MatMulParams *>(param_buf) = mm_params;
                matmul_params = mm_params;
                matmul_params_ready = true;
                matmul_weight = weight;

                args_size = sizeof(MatMulParams);

                if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F16 && activation->type == GGML_TYPE_F32) {
                    if (prefer_rpc) {
                        using fn_type = int(int, int, int, int, int, int, int, int, int);

                        auto op_fn =
                            reinterpret_cast<fn_type *>(dlsym(ops_dl_handle, "htp_ops_rpc_mat_mul_permuted_w16a32"));
                        GGML_ASSERT(op_fn);

                        return op_fn(output_fd, output_offset, activation_fd, activation_offset, weight_fd,
                                     weight_offset, m, k, n);
                    }

                    op_index = HTP_OPS_MAT_MUL_PERMUTED_W16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q4_0 &&
                           activation->type == GGML_TYPE_F32) {
                    // If we skipped host-side permute/repack for this weight, it is in the
                    // original ggml quant layout and must use the "common" decode path.
                    op_index = htp_skip_qkv_permute_repack(weight) ? HTP_OPS_MAT_MUL_COMMON_W4D16A32
                                                                  : HTP_OPS_MAT_MUL_PERMUTED_W4D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_Q8_0 &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = htp_skip_qkv_permute_repack(weight) ? HTP_OPS_MAT_MUL_COMMON_W8D16A32
                                                                  : HTP_OPS_MAT_MUL_PERMUTED_W8D16A32;
                } else if (dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_IQ4_NL &&
                           activation->type == GGML_TYPE_F32) {
                    op_index = htp_skip_qkv_permute_repack(weight) ? HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL
                                                                  : HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL;
                } else {
                    GGML_ASSERT(false && "not implemented");
                }

                if (mm_debug.armed && mm_debug.weight_raw.empty()) {
                    mm_debug.op_index = op_index;
                    mm_debug.weight_raw.resize(ggml_nbytes(weight));
                    std::memcpy(mm_debug.weight_raw.data(), weight->data, mm_debug.weight_raw.size());
                }

                if (htp_trace_matmul_should_log(m)) {
                    // Mirror DSP-side heuristic (see htp-ops-lib/src/dsp/ops/mat_mul.c).
                    const bool use_pipeline = (m >= 128) && (k <= n);
                    const bool use_out_stationary =
                        (weight->type == GGML_TYPE_Q8_0) && (m >= 128) && (k > n) && (n > 1024);
                    const bool is_common_layout =
                        op_index == HTP_OPS_MAT_MUL_COMMON_W4D16A32 ||
                        op_index == HTP_OPS_MAT_MUL_COMMON_W8D16A32 ||
                        op_index == HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL;

                    fprintf(stderr,
                            "HTP_TRACE_MATMUL op=%s dst=%s w=%s(%s) a=%s(%s) m=%d k=%d n=%d "
                            "pipe=%d out_stat=%d common=%d out_off=%lld act_off=%lld w_off=%lld\n",
                            htp_matmul_op_name(op_index),
                            dst->name,
                            (weight->name[0] != '\0') ? weight->name : "<unnamed>", ggml_type_name(weight->type),
                            (activation->name[0] != '\0') ? activation->name : "<unnamed>", ggml_type_name(activation->type),
                            m, k, n,
                            (int) use_pipeline,
                            (int) use_out_stationary,
                            (int) is_common_layout,
                            (long long) output_offset,
                            (long long) activation_offset,
                            (long long) weight_offset);
                }
            }
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            {
                auto * q    = dst->src[0];
                auto * k    = dst->src[1];
                auto * v    = dst->src[2];
                auto * graph_mask = dst->src[3];
                auto * mask       = graph_mask;
                flash_q = q;
                flash_k = k;
                flash_v = v;
                flash_mask = mask;
                flash_scale = *reinterpret_cast<const float *>(&dst->op_params[0]);
                flash_max_bias = *reinterpret_cast<const float *>(&dst->op_params[1]);
                flash_logit_softcap = *reinterpret_cast<const float *>(&dst->op_params[2]);
                const uint32_t flash_flags = *reinterpret_cast<const uint32_t *>(&dst->op_params[4]);
                float flash_kv_scale = *reinterpret_cast<const float *>(&dst->op_params[5]);
                if (!(flash_kv_scale > 0.0f)) {
                    flash_kv_scale = 1.0f;
                }
                const bool flash_prepare_in_kernel = htp_flash_prepare_in_kernel_enabled() && graph_mask == nullptr;
                const bool force_null_mask = htp_flash_force_null_mask_enabled();
                if (force_null_mask) {
                    mask = nullptr;
                }

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == (graph_mask ? 5 : 4));

                auto [o_fd, o_offset] = mappings[0];
                auto [q_fd, q_offset] = mappings[1];
                auto [k_fd, k_offset] = mappings[2];
                auto [v_fd, v_offset] = mappings[3];

                int head_dim   = q->ne[0];
                int qo_len     = q->ne[1];
                int kv_len     = k->ne[1];
                int n_heads    = q->ne[2];
                int n_kv_heads = k->ne[2];
                if (htp_flash_swap_qo_kv_enabled()) {
                    int tmp = qo_len;
                    qo_len = kv_len;
                    kv_len = tmp;
                }

                int mask_fd = -1;
                int mask_offset = 0;
                if (mask) {
                    auto mask_mapping = mappings[4];
                    mask_fd     = mask_mapping.first;
                    mask_offset = (int32_t) mask_mapping.second;
                } else if (!flash_prepare_in_kernel) {
                    auto synthetic = get_or_create_synthetic_flash_mask(qo_len, kv_len);
                    if (synthetic.fd < 0) {
                        if (htp_debug_enabled()) {
                            std::fprintf(stderr,
                                         "htp flash synthetic mask unavailable: qo_len=%d kv_len=%d\n",
                                         qo_len, kv_len);
                        }
                        return -1;
                    }
                    mask_fd = synthetic.fd;
                    mask_offset = 0;
                }

                FlashAttnParams params{
                    .o          = { o_fd,    (int32_t) o_offset    },
                    .q          = { q_fd,    (int32_t) q_offset    },
                    .k          = { k_fd,    (int32_t) k_offset    },
                    .v          = { v_fd,    (int32_t) v_offset    },
                    .mask       = { mask_fd, (int32_t) mask_offset },
                    .qo_len     = qo_len,
                    .kv_len     = kv_len,
                    .n_heads    = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim   = head_dim,
                    .scale      = flash_scale,
                    .kv_scale   = flash_prepare_in_kernel ? flash_kv_scale : 1.0f,
                    .flags      = flash_flags,
                };
                *reinterpret_cast<FlashAttnParams *>(param_buf) = params;

                op_index  = HTP_OPS_FLASH_ATTN_QO_F32_KV_F16;
                args_size = sizeof(FlashAttnParams);

                const int dump_max = htp_flash_dump_max();
                if (dump_max > 0) {
                    static std::atomic<int> dump_counter{ 0 };
                    const int idx = dump_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (idx <= dump_max) {
                        flash_dump_armed = true;
                        flash_dump_idx = idx;
                    }
                }
                if (htp_flash_hash_trace_enabled()) {
                    static std::atomic<int> trace_counter{ 0 };
                    flash_hash_trace_armed = true;
                    flash_hash_trace_idx = trace_counter.fetch_add(1, std::memory_order_relaxed) + 1;
                }
            }
            break;

        case GGML_OP_MAP_CUSTOM2:
            {
                GGML_ASSERT(htp_is_zimg_rope_op(dst));
                GGML_ASSERT(htp_zimg_rope_contract_ok(dst));

                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 3);

                auto [out_fd, out_offset]     = mappings[0];
                auto [src_fd, src_offset]     = mappings[1];
                auto [theta_fd, theta_offset] = mappings[2];

                ZimgRopeParams rope_params{
                    .output  = { out_fd,   (int32_t) out_offset   },
                    .input   = { src_fd,   (int32_t) src_offset   },
                    .theta   = { theta_fd, (int32_t) theta_offset },
                    .d_head  = (int32_t) dst->ne[0],
                    .seq_len = (int32_t) dst->ne[1],
                    .rows    = (int32_t) (dst->ne[2] * dst->ne[3]),
                    .flags   = htp_zimg_rope_flags(dst),
                };
                *reinterpret_cast<ZimgRopeParams *>(param_buf) = rope_params;

                op_index  = HTP_OPS_ZIMG_ROPE_F32;
                args_size = sizeof(ZimgRopeParams);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                if (htp_is_zimg_qknorm_rope_op(dst)) {
                    GGML_ASSERT(htp_zimg_qknorm_rope_contract_ok(dst));
                    auto * src = dst->src[0];
                    auto mappings = get_all_rpcmem_mappings(dst);
                    GGML_ASSERT(mappings.size() == 4);

                    auto [out_fd, out_offset]       = mappings[0];
                    auto [src_fd, src_offset]       = mappings[1];
                    auto [weight_fd, weight_offset] = mappings[2];
                    auto [theta_fd, theta_offset]   = mappings[3];
                    const auto src_nb1 = static_cast<int32_t>(src->nb[1] / sizeof(float));
                    const auto src_nb2 = static_cast<int32_t>(src->nb[2] / sizeof(float));
                    const auto src_nb3 = static_cast<int32_t>(src->nb[3] / sizeof(float));
                    const auto theta_start = static_cast<int32_t>(htp_zimg_qknorm_rope_theta_start(dst));

                    ZimgQkNormRopeParams params{
                        .output         = { out_fd,    (int32_t) out_offset    },
                        .input          = { src_fd,    (int32_t) src_offset    },
                        .weight         = { weight_fd, (int32_t) weight_offset },
                        .theta          = { theta_fd,  (int32_t) theta_offset  },
                        .d_head         = (int32_t) dst->ne[0],
                        .seq_len        = (int32_t) dst->ne[1],
                        .rows           = (int32_t) (dst->ne[2] * dst->ne[3]),
                        .rows_per_batch = (int32_t) dst->ne[2],
                        .src_nb1        = src_nb1,
                        .src_nb2        = src_nb2,
                        .src_nb3        = src_nb3,
                        .theta_start    = theta_start,
                        .flags          = htp_zimg_qknorm_rope_flags(dst),
                    };
                    *reinterpret_cast<ZimgQkNormRopeParams *>(param_buf) = params;

                    op_index  = HTP_OPS_ZIMG_QKNORM_ROPE_F32;
                    args_size = sizeof(ZimgQkNormRopeParams);
                    break;
                }

                GGML_ASSERT(htp_is_flux_ss_linear2_fused_op(dst));
                GGML_ASSERT(htp_flux_ss_linear2_contract_ok(dst));

                auto * attn   = dst->src[0];
                auto * mlp    = dst->src[1];
                auto * weight = dst->src[2];

                GGML_ASSERT(htp_repack_quant_weight_inplace_if_needed(weight));
                auto mappings = get_all_rpcmem_mappings(dst);
                GGML_ASSERT(mappings.size() == 4);

                auto [out_fd, out_offset]         = mappings[0];
                auto [attn_fd, attn_offset]       = mappings[1];
                auto [mlp_fd, mlp_offset]         = mappings[2];
                auto [weight_fd, weight_offset]   = mappings[3];

                FluxSingleStreamLinear2Params params{
                    .output = { out_fd,    (int32_t) out_offset    },
                    .attn   = { attn_fd,   (int32_t) attn_offset   },
                    .mlp    = { mlp_fd,    (int32_t) mlp_offset    },
                    .weight = { weight_fd, (int32_t) weight_offset },
                    .m      = (int32_t) ggml_nrows(attn),
                    .attn_k = (int32_t) attn->ne[0],
                    .mlp_k  = (int32_t) mlp->ne[0],
                    .n      = (int32_t) dst->ne[0],
                    .flags  = 0u,
                };
                *reinterpret_cast<FluxSingleStreamLinear2Params *>(param_buf) = params;

                op_index  = HTP_OPS_FLUX_SS_LINEAR2_FUSED_Q8;
                args_size = sizeof(FluxSingleStreamLinear2Params);
            }
            break;

        default:
            break;
    }

    int state = 0;
    bool dispatched = false;
    const bool is_matmul_op =
        op_index == HTP_OPS_MAT_MUL_PERMUTED_W16A32 ||
        op_index == HTP_OPS_MAT_MUL_PERMUTED_W4D16A32 ||
        op_index == HTP_OPS_MAT_MUL_PERMUTED_W8D16A32 ||
        op_index == HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL ||
        op_index == HTP_OPS_MAT_MUL_COMMON_W4D16A32 ||
        op_index == HTP_OPS_MAT_MUL_COMMON_W8D16A32 ||
        op_index == HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL;

    if (is_matmul_op && matmul_params_ready) {
        const int split_m = htp_matmul_split_m_for_weight(matmul_weight);
        if (split_m > 0 && matmul_params.m > split_m) {
            const int m = matmul_params.m;
            const int k = matmul_params.k;
            const int n = matmul_params.n;
            const int32_t out_off_base = matmul_params.output.offset;
            const int32_t act_off_base = matmul_params.activation.offset;
            const int64_t out_row_bytes = (int64_t) n * (int64_t) sizeof(float);
            const int64_t act_row_bytes = (int64_t) k * (int64_t) sizeof(float);
            for (int row0 = 0; row0 < m; row0 += split_m) {
                MatMulParams chunk = matmul_params;
                chunk.m = std::min(split_m, m - row0);
                const int64_t out_off = (int64_t) out_off_base + (int64_t) row0 * out_row_bytes;
                const int64_t act_off = (int64_t) act_off_base + (int64_t) row0 * act_row_bytes;
                if (out_off > INT32_MAX || act_off > INT32_MAX) {
                    if (htp_debug_enabled()) {
                        fprintf(stderr,
                                "matmul split-m overflow: m=%d split=%d row0=%d out_off=%lld act_off=%lld\n",
                                m, split_m, row0, (long long) out_off, (long long) act_off);
                    }
                    state = -1;
                    break;
                }
                chunk.output.offset = (int32_t) out_off;
                chunk.activation.offset = (int32_t) act_off;
                state = htp_ops_issue_request(ctx, op_index, &chunk, sizeof(chunk));
                if (state != 0) {
                    break;
                }
            }
            dispatched = true;
            if (htp_debug_enabled()) {
                fprintf(stderr, "HTP matmul split-m dispatched: m=%d split=%d k=%d n=%d w_name=%s state=%d\n",
                        m, split_m, k, n,
                        (matmul_weight && matmul_weight->name[0] != '\0') ? matmul_weight->name : "<unnamed>",
                        state);
            }
        }
    }

    if (!dispatched) {
        state = htp_ops_issue_request(ctx, op_index, param_buf, args_size);
    }

    if (state == 0 && is_matmul_op) {
        auto & mm_debug = htp_matmul_debug_snapshot();
        if (htp_matmul_debug_check_enabled() && mm_debug.armed && !mm_debug.done) {
            htp_debug_compare_matmul_snapshot(mm_debug);
            if (htp_matmul_debug_raw_dump_only_selected(mm_debug.call_idx)) {
                std::fprintf(stderr,
                             "HTP_MATMUL_DEBUG raw dump emitted for call=%llu, exiting before graph continues\n",
                             (unsigned long long) mm_debug.call_idx);
                std::fflush(stderr);
                std::exit(0);
            }
            mm_debug.done = !htp_matmul_debug_check_all_enabled();
            mm_debug.armed = false;
            mm_debug.weight_name.clear();
            mm_debug.weight_raw.clear();
        }
    }

    if (state == 0 && flash_dump_armed && flash_q != nullptr && flash_k != nullptr && flash_v != nullptr) {
        htp_flash_dump_tensors(flash_dump_idx,
                               flash_q,
                               flash_k,
                               flash_v,
                               flash_mask,
                               dst,
                               flash_scale,
                               flash_max_bias,
                               flash_logit_softcap,
                               *reinterpret_cast<const uint32_t *>(&dst->op_params[4]),
                               *reinterpret_cast<const float *>(&dst->op_params[5]));
    }

    if (state == 0 && flash_hash_trace_armed && flash_q != nullptr && flash_k != nullptr && flash_v != nullptr) {
        std::fprintf(stderr,
                     "HTP_FLASH_HASH idx=%d qo=%ld kv=%ld heads=%ld kv_heads=%ld dim=%ld q=%016llx k=%016llx "
                     "v=%016llx mask=%016llx o=%016llx scale=%g kv_scale=%g\n",
                     flash_hash_trace_idx,
                     flash_q->ne[1], flash_k->ne[1], flash_q->ne[2], flash_k->ne[2], flash_q->ne[0],
                     (unsigned long long) htp_tensor_hash64(flash_q),
                     (unsigned long long) htp_tensor_hash64(flash_k),
                     (unsigned long long) htp_tensor_hash64(flash_v),
                     (unsigned long long) htp_tensor_hash64(flash_mask),
                     (unsigned long long) htp_tensor_hash64(dst),
                     flash_scale,
                     *reinterpret_cast<const float *>(&dst->op_params[5]));
    }

    if (act_override_active) {
        htp_release_activation_override(act_override);
    }

    return state;
}

bool ggml_htp_try_compute(struct ggml_compute_params * params, struct ggml_tensor * dst) {
    htp_last_attempt_reset();

    if (!htp_ops_support_op(dst)) {
        return false;
    }

    g_htp_last_attempt_state.copy_bytes = htp_estimate_copy_bytes(dst);

    // NOTE: ggml calls this hook from every worker thread. We dispatch once on ith==0, but
    // all threads must agree on whether the op was offloaded or should fall back to CPU.
    static std::atomic<int> s_last_op_rc{ 0 };

    int op_rc = 0;
    if (params->ith == 0) {
        htp_op_stats_record_op(dst->op);
        auto t0 = std::chrono::steady_clock::now();
        op_rc = htp_ops_compute_op(params, dst);
        auto t1 = std::chrono::steady_clock::now();
        htp_op_stats_record_latency(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        s_last_op_rc.store(op_rc, std::memory_order_release);
        if (op_rc != 0 && htp_debug_enabled()) {
            fprintf(stderr, "htp op failed: op=%s rc=%d dst=%s\n", ggml_op_name(dst->op), op_rc, dst->name);
        }
    }

    // Synchronize with ith==0 so all threads see the same op_rc decision.
    if (params->nth > 1 && params->threadpool != nullptr) {
        ggml_barrier(params->threadpool);
        op_rc = s_last_op_rc.load(std::memory_order_acquire);
    }

    if (op_rc != 0) {
        if (params->ith == 0) {
            htp_fallback_record(HtpFallbackReason::kComputeFailureSingleThread, dst);
        }
        return false;
    }
    return true;
}
}

extern "C" const char * ggml_htp_take_last_fallback_reason(void) {
    HtpFallbackReason reason = g_htp_last_attempt_state.fallback_reason;
    g_htp_last_attempt_state.fallback_reason = HtpFallbackReason::kCount;
    if (reason == HtpFallbackReason::kCount) {
        return nullptr;
    }
    return htp_fallback_reason_name(reason);
}

extern "C" uint64_t ggml_htp_take_last_copy_bytes(void) {
    const uint64_t copy_bytes = g_htp_last_attempt_state.copy_bytes;
    g_htp_last_attempt_state.copy_bytes = 0;
    return copy_bytes;
}
