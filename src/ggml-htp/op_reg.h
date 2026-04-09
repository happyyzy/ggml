#pragma once

#include <stdint.h>

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_WF8A16,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  // "Common" layout: keep GGUF weights in their original ggml quant layout and
  // let DSP dequantize + scatter into HMX tile layout on-chip (no host-side
  // permute+requant drift).
  HTP_OPS_MAT_MUL_COMMON_W4D16A32,
  HTP_OPS_MAT_MUL_COMMON_W8D16A32,
  HTP_OPS_MAT_MUL_COMMON_W4D16A32_IQ4_NL,
  HTP_OPS_ZIMG_ROPE_F32,
  HTP_OPS_ZIMG_QKNORM_ROPE_F32,
  HTP_OPS_FLUX_SS_LINEAR2_FUSED_Q8,
  HTP_OPS_COUNT,
};

enum HtpZimgRopeFlags {
  HTP_ZIMG_ROPE_FLAG_INTERLEAVED = 1u << 0,
  HTP_ZIMG_QKNORM_ROPE_FLAG_FORCE_SCALAR_RMSNORM = 1u << 16,
  HTP_ZIMG_QKNORM_ROPE_FLAG_FORCE_SCALAR_MUL     = 1u << 17,
  HTP_ZIMG_QKNORM_ROPE_FLAG_FORCE_SCALAR_ROPE    = 1u << 18,
};

enum HtpMatMulFlags {
  HTP_MATMUL_FLAG_Q8_OUT_STATIONARY = 1u << 0,
  HTP_MATMUL_FLAG_DBG_OUTSTAT_ACT_TILE_DUMP = 1u << 1,
  HTP_MATMUL_FLAG_DBG_ACT_DIRECT_STAGE = 1u << 2,
  HTP_MATMUL_FLAG_DBG_ACT_SCRATCH_DIRECT_TRANSFER = 1u << 3,
  HTP_MATMUL_FLAG_DBG_OUTSTAT_SCRATCH_HVX_DUMP = 1u << 4,
  HTP_MATMUL_FLAG_DBG_ACT_META_OUT = 1u << 22,
  HTP_MATMUL_FLAG_DBG_OUTSTAT_FIRSTBLOCK_DUMP = 1u << 31,
};

enum HtpWf8MatMulFlags {
  HTP_WF8_MATMUL_FLAG_FORCE_SIMPLE  = 1u << 0,
  HTP_WF8_MATMUL_FLAG_FORCE_OUTSTAT = 1u << 1,
  HTP_WF8_MATMUL_FLAG_FORCE_PIPE    = 1u << 2,
};

#define MATMUL_FLAG_FORCE_SIMPLE  HTP_WF8_MATMUL_FLAG_FORCE_SIMPLE
#define MATMUL_FLAG_FORCE_OUTSTAT HTP_WF8_MATMUL_FLAG_FORCE_OUTSTAT
#define MATMUL_FLAG_FORCE_PIPE    HTP_WF8_MATMUL_FLAG_FORCE_PIPE

struct RpcmemBufAddr {
  int32_t fd;
  int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));

struct MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k
  struct RpcmemBufAddr weight; // k * n
  int32_t m;
  int32_t k;
  int32_t n;
  uint32_t flags;
} __attribute__((packed));

struct FlashAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
  float   scale;
  float   kv_scale;
  uint32_t flags;
} __attribute__((packed));

struct ZimgRopeParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr input;
  struct RpcmemBufAddr theta;
  int32_t  d_head;
  int32_t  seq_len;
  int32_t  rows;
  uint32_t flags;
} __attribute__((packed));

struct ZimgQkNormRopeParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr input;
  struct RpcmemBufAddr weight;
  struct RpcmemBufAddr theta;
  int32_t  d_head;
  int32_t  seq_len;
  int32_t  rows;
  int32_t  rows_per_batch;
  int32_t  src_nb1;
  int32_t  src_nb2;
  int32_t  src_nb3;
  int32_t  theta_start;
  uint32_t flags;
} __attribute__((packed));

struct FluxSingleStreamLinear2Params {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr attn;
  struct RpcmemBufAddr mlp;
  struct RpcmemBufAddr weight;
  int32_t m;
  int32_t attn_k;
  int32_t mlp_k;
  int32_t n;
  uint32_t flags;
} __attribute__((packed));
