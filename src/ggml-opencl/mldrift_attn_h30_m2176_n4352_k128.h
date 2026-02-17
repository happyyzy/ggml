// Auto-extracted from replay_attention_attn_f32_b1_h30_m2176_n4352_k128.cc
#pragma once

#include <CL/cl.h>

namespace ggml_opencl_mldrift_h30_m2176_n4352 {

static inline bool mldrift_cl_check(cl_int status, const char * msg) {
    if (status != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl: mldrift %s failed (%d)\n", msg, (int) status);
        return false;
    }
    return true;
}

static const char* kProgramSource_0 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int S = get_global_id(2);
  if (X >= shared_int4_0.w || Y >= shared_int4_0.y || S >= shared_int4_0.z) { 
    return; 
  } 
  half temps[4];
  temps[0] = (half)(0.f);
  temps[1] = (half)(0.f);
  temps[2] = (half)(0.f);
  temps[3] = (half)(0.f);
  for (int i = 0; i < 4; ++i) {
    int dst_channel = S * 4 + i;
    if (dst_channel < shared_int4_0.x) {
      int s_y = Y;
      int s_x = dst_channel;
      int s_c = X;
        {
  int slice_coord_TMP = (s_c) / 4;
  int sub_ch_coord_TMP = (s_c) % 4;
  half4 src_TMP = read_imageh(src_tensor_image2d, smp_zero, (int2)((s_x), ((s_y) * shared_int4_1.x + (slice_coord_TMP))));
  temps[i] = (half[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];
  };
    }
  }
  half4 result;
  result.x = temps[0];
  result.y = temps[1];
  result.z = temps[2];
  result.w = temps[3];
  dst_tensor_buffer[(((S) * shared_int4_0.y + (Y)) * shared_int4_0.w + (X))] = result;
}
)CLC";

static const char* kProgramSource_1 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __read_only image1d_buffer_t src_image_buffer,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2,
  half4 shared_half4_0) {
  int linear_index = get_global_id(0);
  if (linear_index >= shared_int4_0.y) return;
  if (get_global_id(1) != 0) return;
  if (get_global_id(2) != 0) return;
  int dst_o_sp_i_ogroup = linear_index;
  int dst_ogroup = dst_o_sp_i_ogroup % shared_int4_0.x;
  int dst_o_sp_i = dst_o_sp_i_ogroup / shared_int4_0.x;
  int dst_i = dst_o_sp_i % shared_int4_0.z;
  int dst_o_sp = dst_o_sp_i / shared_int4_0.z;
  int dst_sp = dst_o_sp % shared_int4_1.w;
  int dst_o = dst_o_sp / shared_int4_1.w;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = dst_sp;
  int W = spatial_linear % shared_int4_2.x;
  int H = spatial_linear / shared_int4_2.x;
  half4 w0 = (half4)(0);
  half4 w1 = (half4)(0);
  half4 w2 = (half4)(0);
  half4 w3 = (half4)(0);

  if (i_slice * 4 < shared_int4_1.z && o_slice < shared_int4_1.x) {
    w0 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_0.w + (W)) * shared_int4_1.y + (i_slice * 4)));
  }
  if (i_slice * 4 + 1 < shared_int4_1.z && o_slice < shared_int4_1.x) {
    w1 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_0.w + (W)) * shared_int4_1.y + (i_slice * 4 + 1)));
  }
  if (i_slice * 4 + 2 < shared_int4_1.z && o_slice < shared_int4_1.x) {
    w2 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_0.w + (W)) * shared_int4_1.y + (i_slice * 4 + 2)));
  }
  if (i_slice * 4 + 3 < shared_int4_1.z && o_slice < shared_int4_1.x) {
    w3 = read_imageh(src_image_buffer, (((o_slice) * shared_int4_0.w + (W)) * shared_int4_1.y + (i_slice * 4 + 3)));
  }
  if (o_slice == shared_int4_1.x - 1) {
    half4 mask = (half4)(shared_half4_0.y, shared_half4_0.z, shared_half4_0.w, shared_half4_0.x);
    w0 *= mask;
    w1 *= mask;
    w2 *= mask;
    w3 *= mask;
  }
  half4 r0 = w0;
  half4 r1 = w1;
  half4 r2 = w2;
  half4 r3 = w3;
  dst_tensor_buffer[linear_index * 4 + 0] = r0;
  dst_tensor_buffer[linear_index * 4 + 1] = r1;
  dst_tensor_buffer[linear_index * 4 + 2] = r2;
  dst_tensor_buffer[linear_index * 4 + 3] = r3;
}
)CLC";

static const char* kProgramSource_2 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
__attribute__((qcom_max_concurrent_subgroups(12)))
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __read_only image2d_t src_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.f);
  half4 r1 = (half4)(0.f);
  half4 r2 = (half4)(0.f);
  half4 r3 = (half4)(0.f);
  half4 r4 = (half4)(0.f);
  half4 r5 = (half4)(0.f);
  half4 r6 = (half4)(0.f);
  half4 r7 = (half4)(0.f);

  int x_coord = mad24(X, shared_int4_2.y, shared_int4_1.y);
  int y_coord = mad24(Y, shared_int4_2.z, shared_int4_1.z);
  int coord_x, coord_y, coord_s;
  int f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  __constant half16* weights_cache = (__constant half16*)&xmem_buffer[c_offset];

  coord_y = Y;
  coord_x = X;
      coord_s = 0;
      do {
        half4 src0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_2.x + (coord_s))));
; coord_s++;
        half4 src1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((coord_x), ((coord_y) * shared_int4_2.x + (coord_s))));
; coord_s++;
        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
        f_offset += 64;
        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
  r0 += src0.x * weights_cache[0].s0123;
  r0 += src0.y * weights_cache[0].s4567;
  r0 += src0.z * weights_cache[0].s89ab;
  r0 += src0.w * weights_cache[0].scdef;
  r1 += src0.x * weights_cache[1].s0123;
  r1 += src0.y * weights_cache[1].s4567;
  r1 += src0.z * weights_cache[1].s89ab;
  r1 += src0.w * weights_cache[1].scdef;
  r2 += src0.x * weights_cache[2].s0123;
  r2 += src0.y * weights_cache[2].s4567;
  r2 += src0.z * weights_cache[2].s89ab;
  r2 += src0.w * weights_cache[2].scdef;
  r3 += src0.x * weights_cache[3].s0123;
  r3 += src0.y * weights_cache[3].s4567;
  r3 += src0.z * weights_cache[3].s89ab;
  r3 += src0.w * weights_cache[3].scdef;
  r4 += src0.x * weights_cache[4].s0123;
  r4 += src0.y * weights_cache[4].s4567;
  r4 += src0.z * weights_cache[4].s89ab;
  r4 += src0.w * weights_cache[4].scdef;
  r5 += src0.x * weights_cache[5].s0123;
  r5 += src0.y * weights_cache[5].s4567;
  r5 += src0.z * weights_cache[5].s89ab;
  r5 += src0.w * weights_cache[5].scdef;
  r6 += src0.x * weights_cache[6].s0123;
  r6 += src0.y * weights_cache[6].s4567;
  r6 += src0.z * weights_cache[6].s89ab;
  r6 += src0.w * weights_cache[6].scdef;
  r7 += src0.x * weights_cache[7].s0123;
  r7 += src0.y * weights_cache[7].s4567;
  r7 += src0.z * weights_cache[7].s89ab;
  r7 += src0.w * weights_cache[7].scdef;
  r0 += src1.x * weights_cache[8].s0123;
  r0 += src1.y * weights_cache[8].s4567;
  r0 += src1.z * weights_cache[8].s89ab;
  r0 += src1.w * weights_cache[8].scdef;
  r1 += src1.x * weights_cache[9].s0123;
  r1 += src1.y * weights_cache[9].s4567;
  r1 += src1.z * weights_cache[9].s89ab;
  r1 += src1.w * weights_cache[9].scdef;
  r2 += src1.x * weights_cache[10].s0123;
  r2 += src1.y * weights_cache[10].s4567;
  r2 += src1.z * weights_cache[10].s89ab;
  r2 += src1.w * weights_cache[10].scdef;
  r3 += src1.x * weights_cache[11].s0123;
  r3 += src1.y * weights_cache[11].s4567;
  r3 += src1.z * weights_cache[11].s89ab;
  r3 += src1.w * weights_cache[11].scdef;
  r4 += src1.x * weights_cache[12].s0123;
  r4 += src1.y * weights_cache[12].s4567;
  r4 += src1.z * weights_cache[12].s89ab;
  r4 += src1.w * weights_cache[12].scdef;
  r5 += src1.x * weights_cache[13].s0123;
  r5 += src1.y * weights_cache[13].s4567;
  r5 += src1.z * weights_cache[13].s89ab;
  r5 += src1.w * weights_cache[13].scdef;
  r6 += src1.x * weights_cache[14].s0123;
  r6 += src1.y * weights_cache[14].s4567;
  r6 += src1.z * weights_cache[14].s89ab;
  r6 += src1.w * weights_cache[14].scdef;
  r7 += src1.x * weights_cache[15].s0123;
  r7 += src1.y * weights_cache[15].s4567;
  r7 += src1.z * weights_cache[15].s89ab;
  r7 += src1.w * weights_cache[15].scdef;
      } while (coord_s < shared_int4_2.x);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r0);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r1);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r2);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r3);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r4);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r5);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r6);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r7);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image2d, smp_zero, (int2)((0), ((0) * shared_int4_2.x + (0))));
    }
    dst_tensor_buffer[(((coord_s) * shared_int4_0.x + (coord_y)) * shared_int4_0.z + (coord_x))] = res;
    coord_s++;
  }
}
)CLC";

static const char* kProgramSource_3 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__read_only image1d_buffer_t src_tensor_image_buffer,
  __write_only image2d_t dst_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return; 
  float sum = 0.0f;
  int end_channel = shared_int4_0.w;
  int end_slice = (end_channel + 3) / 4;
  int start_channel = 0;
  int start_slice = start_channel / 4;
  bool need_per_channels_check = start_channel % 4 != 0 || end_channel % 4 != 0;
  float maximum;
    {
  int slice_coord_TMP = (start_channel) / 4;
  int sub_ch_coord_TMP = (start_channel) % 4;
  float4 src_TMP = read_imagef(src_tensor_image_buffer, (((slice_coord_TMP) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
  maximum = (float[4]){src_TMP.x, src_TMP.y, src_TMP.z, src_TMP.w}[sub_ch_coord_TMP];
  };
  for (int d = start_slice; d < end_slice; d += 1) {
    float4 mask_dot = (float4)(1.0f);
    float4 src = read_imagef(src_tensor_image_buffer, (((d) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
    if (need_per_channels_check && (d == start_slice || d == end_slice - 1)) {
      if (d * 4 + 0 < start_channel || d * 4 + 0 >= end_channel) {
        mask_dot.x = 0.0f;
        src.x = maximum;
      }
      if (d * 4 + 1 < start_channel || d * 4 + 1 >= end_channel) {
        mask_dot.y = 0.0f;
        src.y = maximum;
      }
      if (d * 4 + 2 < start_channel || d * 4 + 2 >= end_channel) {
        mask_dot.z = 0.0f;
        src.z = maximum;
      }
      if (d * 4 + 3 < start_channel || d * 4 + 3 >= end_channel) {
        mask_dot.w = 0.0f;
        src.w = maximum;
      }
    }
    float new_max = max(src.x, src.y);
    new_max = max(new_max, src.z);
    new_max = max(new_max, src.w);
    new_max = max(new_max, maximum);
    float scale = native_exp(maximum - new_max);
    maximum = new_max;
    sum *= scale;
    float4 exp_res = native_exp(src - (float4)(maximum));
    sum += dot(mask_dot, exp_res);
  }
  float inv_sum = 1.0f / sum;
  half4 result;
  result.x = convert_half(inv_sum);
  result.y = convert_half(maximum);
  write_imageh(dst_tensor_image2d, (int2)((X), ((Y) * shared_int4_0.y + (0))), result);
}
)CLC";

static const char* kProgramSource_4 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __read_only image1d_buffer_t src_tensor_image_buffer,
  __read_only image2d_t src_tensor_1_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x || Z >= shared_int4_0.y) return; 
  half4 src = read_imageh(src_tensor_image_buffer, (((Z) * shared_int4_1.x + (Y)) * shared_int4_1.y + (X)));
  {

   half4 src_final;
  {  
  {  half4 exp_val = read_imageh(src_tensor_1_image2d, smp_zero, (int2)(((X)), (((Y)) * shared_int4_0.w + (0))));
    src_final = exp(src - exp_val.y) * exp_val.x;
  }
  }
  dst_tensor_buffer[(((Z) * shared_int4_0.x + (Y)) * shared_int4_0.z + (X))] = src_final;
};
} 
)CLC";

static const char* kProgramSource_5 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global half4* dst_tensor_buffer,
  __read_only image2d_t src_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  half4 shared_half4_0) {
  int linear_index = get_global_id(0);
  if (linear_index >= shared_int4_0.y) return;
  if (get_global_id(1) != 0) return;
  if (get_global_id(2) != 0) return;
  int dst_o_sp_i_ogroup = linear_index;
  int dst_ogroup = dst_o_sp_i_ogroup % shared_int4_0.x;
  int dst_o_sp_i = dst_o_sp_i_ogroup / shared_int4_0.x;
  int dst_i = dst_o_sp_i % shared_int4_0.z;
  int dst_o_sp = dst_o_sp_i / shared_int4_0.z;
  int dst_sp = dst_o_sp % shared_int4_1.y;
  int dst_o = dst_o_sp / shared_int4_1.y;
  int i_slice = dst_i;
  int o_slice = dst_o * shared_int4_0.x + dst_ogroup;
  int spatial_linear = dst_sp;
  int W = spatial_linear % shared_int4_1.z;
  int H = spatial_linear / shared_int4_1.z;
  half4 w0 = (half4)(0);
  half4 w1 = (half4)(0);
  half4 w2 = (half4)(0);
  half4 w3 = (half4)(0);

  if (i_slice * 4 < shared_int4_1.x && o_slice < shared_int4_0.w) {
    w0 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4), ((W) * shared_int4_0.w + (o_slice))));
  }
  if (i_slice * 4 + 1 < shared_int4_1.x && o_slice < shared_int4_0.w) {
    w1 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 1), ((W) * shared_int4_0.w + (o_slice))));
  }
  if (i_slice * 4 + 2 < shared_int4_1.x && o_slice < shared_int4_0.w) {
    w2 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 2), ((W) * shared_int4_0.w + (o_slice))));
  }
  if (i_slice * 4 + 3 < shared_int4_1.x && o_slice < shared_int4_0.w) {
    w3 = read_imageh(src_image2d, smp_zero, (int2)((i_slice * 4 + 3), ((W) * shared_int4_0.w + (o_slice))));
  }
  if (o_slice == shared_int4_0.w - 1) {
    half4 mask = (half4)(shared_half4_0.y, shared_half4_0.z, shared_half4_0.w, shared_half4_0.x);
    w0 *= mask;
    w1 *= mask;
    w2 *= mask;
    w3 *= mask;
  }
  half4 r0 = w0;
  half4 r1 = w1;
  half4 r2 = w2;
  half4 r3 = w3;
  dst_tensor_buffer[linear_index * 4 + 0] = r0;
  dst_tensor_buffer[linear_index * 4 + 1] = r1;
  dst_tensor_buffer[linear_index * 4 + 2] = r2;
  dst_tensor_buffer[linear_index * 4 + 3] = r3;
}
)CLC";

static const char* kProgramSource_6 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_qcom_subgroup_uniform_load: enable
#pragma OPENCL EXTENSION cl_qcom_subgroup_constant_load: enable
__attribute__((qcom_max_concurrent_subgroups(12)))
MAIN_FUNCTION(__constant half8* weights_buffer  __attribute__((sub_group_uniform)),
  __constant half8* xmem_buffer  __attribute__((max_constant_size((6144)))),
  __read_only image1d_buffer_t src_tensor_image_buffer,
  __write_only image2d_t dst_tensor_image2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2,
  int4 shared_int4_3) {
  int X = get_group_id(1) * get_local_size(0) + get_local_id(0);
  int Y = get_group_id(2) * get_local_size(1) + get_local_id(1);
  int Z = get_group_id(0) * get_local_size(2) + get_local_id(2);
  if (X >= shared_int4_0.z || Y >= shared_int4_0.x) return;
  if (Z * 8 >= shared_int4_0.y) return;

  half4 r0 = (half4)(0.f);
  half4 r1 = (half4)(0.f);
  half4 r2 = (half4)(0.f);
  half4 r3 = (half4)(0.f);
  half4 r4 = (half4)(0.f);
  half4 r5 = (half4)(0.f);
  half4 r6 = (half4)(0.f);
  half4 r7 = (half4)(0.f);

  int x_coord = mad24(X, shared_int4_2.w, shared_int4_1.y);
  int y_coord = mad24(Y, shared_int4_3.x, shared_int4_1.z);
  int coord_x, coord_y, coord_s;
  int f_offset = (Z * shared_int4_1.w + Y) * shared_int4_1.x * 32;

  int subgroup_id = (int)((0x1F & qcom_get_physical_sub_group_id()));
  subgroup_id = subgroup_id % 12;
  int c_offset = mul24(subgroup_id, shared_int4_0.w);
  __constant half16* weights_cache = (__constant half16*)&xmem_buffer[c_offset];

  coord_y = Y;
  coord_x = X;
      int addr = (((0) * shared_int4_1.w + (coord_y)) * shared_int4_2.z + (coord_x));
      int dz = shared_int4_2.x;
      coord_s = 0;
      do {
        half4 src0 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
; coord_s++;
        half4 src1 = read_imageh(src_tensor_image_buffer, addr); addr += dz;
; coord_s++;
        qcom_sub_group_constant_load8(xmem_buffer, weights_buffer, c_offset, f_offset >> 1, 32);
        f_offset += 64;
        qcom_sub_group_sync(QCOM_CLK_CONST_LOAD_SYNC);
  r0 += src0.x * weights_cache[0].s0123;
  r0 += src0.y * weights_cache[0].s4567;
  r0 += src0.z * weights_cache[0].s89ab;
  r0 += src0.w * weights_cache[0].scdef;
  r1 += src0.x * weights_cache[1].s0123;
  r1 += src0.y * weights_cache[1].s4567;
  r1 += src0.z * weights_cache[1].s89ab;
  r1 += src0.w * weights_cache[1].scdef;
  r2 += src0.x * weights_cache[2].s0123;
  r2 += src0.y * weights_cache[2].s4567;
  r2 += src0.z * weights_cache[2].s89ab;
  r2 += src0.w * weights_cache[2].scdef;
  r3 += src0.x * weights_cache[3].s0123;
  r3 += src0.y * weights_cache[3].s4567;
  r3 += src0.z * weights_cache[3].s89ab;
  r3 += src0.w * weights_cache[3].scdef;
  r4 += src0.x * weights_cache[4].s0123;
  r4 += src0.y * weights_cache[4].s4567;
  r4 += src0.z * weights_cache[4].s89ab;
  r4 += src0.w * weights_cache[4].scdef;
  r5 += src0.x * weights_cache[5].s0123;
  r5 += src0.y * weights_cache[5].s4567;
  r5 += src0.z * weights_cache[5].s89ab;
  r5 += src0.w * weights_cache[5].scdef;
  r6 += src0.x * weights_cache[6].s0123;
  r6 += src0.y * weights_cache[6].s4567;
  r6 += src0.z * weights_cache[6].s89ab;
  r6 += src0.w * weights_cache[6].scdef;
  r7 += src0.x * weights_cache[7].s0123;
  r7 += src0.y * weights_cache[7].s4567;
  r7 += src0.z * weights_cache[7].s89ab;
  r7 += src0.w * weights_cache[7].scdef;
  r0 += src1.x * weights_cache[8].s0123;
  r0 += src1.y * weights_cache[8].s4567;
  r0 += src1.z * weights_cache[8].s89ab;
  r0 += src1.w * weights_cache[8].scdef;
  r1 += src1.x * weights_cache[9].s0123;
  r1 += src1.y * weights_cache[9].s4567;
  r1 += src1.z * weights_cache[9].s89ab;
  r1 += src1.w * weights_cache[9].scdef;
  r2 += src1.x * weights_cache[10].s0123;
  r2 += src1.y * weights_cache[10].s4567;
  r2 += src1.z * weights_cache[10].s89ab;
  r2 += src1.w * weights_cache[10].scdef;
  r3 += src1.x * weights_cache[11].s0123;
  r3 += src1.y * weights_cache[11].s4567;
  r3 += src1.z * weights_cache[11].s89ab;
  r3 += src1.w * weights_cache[11].scdef;
  r4 += src1.x * weights_cache[12].s0123;
  r4 += src1.y * weights_cache[12].s4567;
  r4 += src1.z * weights_cache[12].s89ab;
  r4 += src1.w * weights_cache[12].scdef;
  r5 += src1.x * weights_cache[13].s0123;
  r5 += src1.y * weights_cache[13].s4567;
  r5 += src1.z * weights_cache[13].s89ab;
  r5 += src1.w * weights_cache[13].scdef;
  r6 += src1.x * weights_cache[14].s0123;
  r6 += src1.y * weights_cache[14].s4567;
  r6 += src1.z * weights_cache[14].s89ab;
  r6 += src1.w * weights_cache[14].scdef;
  r7 += src1.x * weights_cache[15].s0123;
  r7 += src1.y * weights_cache[15].s4567;
  r7 += src1.z * weights_cache[15].s89ab;
  r7 += src1.w * weights_cache[15].scdef;
      } while (coord_s < shared_int4_2.y);

  coord_s = mul24(Z, 8);
  coord_x = X;
  coord_y = Y;
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r0);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r1);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r2);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r3);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r4);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r5);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r6);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
  if (coord_s < shared_int4_0.y) {
    half4 res = convert_half4(r7);
    if (coord_s < 0) {
      res += read_imageh(src_tensor_image_buffer, (((0) * shared_int4_1.w + (0)) * shared_int4_2.z + (0)));
    }
    write_imageh(dst_tensor_image2d, (int2)((coord_x), ((coord_y) * shared_int4_0.y + (coord_s))), res);
    coord_s++;
  }
}
)CLC";

static const char* kProgramSource_7 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global float* buffer_buffer,
  __write_only image2d_t tensor_image2d,
  int4 shared_int4_0) {
  int linear_id = get_global_id(0);
  int x = linear_id / 1;
  int b = linear_id % 1;
  int y = get_global_id(1);
  int d = get_global_id(2);

  if (x >= shared_int4_0.w || y >= shared_int4_0.y || d >= shared_int4_0.z) return;
  int c = d * 4;
  int index = ((b * shared_int4_0.y + y) * shared_int4_0.w + x) * shared_int4_0.x + c;
  half4 result = (half4)(0.f);
  result.x = convert_half(buffer_buffer[index]);
  if (c + 1 < shared_int4_0.x) {
    result.y = convert_half(buffer_buffer[index + 1]);
  }
  if (c + 2 < shared_int4_0.x) {
    result.z = convert_half(buffer_buffer[index + 2]);
  }
  if (c + 3 < shared_int4_0.x) {
    result.w = convert_half(buffer_buffer[index + 3]);
  }
  write_imageh(tensor_image2d, (int2)((x), ((y) * shared_int4_0.z + (d))), result);
}
)CLC";

static const char* kProgramSource_8 = R"CLC(
#define MAIN_FUNCTION __kernel void main_function
#define bool2 uchar2
#define bool3 uchar3
#define bool4 uchar4
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
MAIN_FUNCTION(__global float* buffer_buffer,
  __read_only image2d_t tensor_image2d,
  int4 shared_int4_0) {
  int linear_id = get_global_id(0);
  int x = linear_id / 1;
  int b = linear_id % 1;
  int y = get_global_id(1);
  int d = get_global_id(2);
  if (x >= shared_int4_0.w || y >= shared_int4_0.y || d >= shared_int4_0.z) return;
  half4 in_value = read_imageh(tensor_image2d, smp_zero, (int2)((x), ((y) * shared_int4_0.z + (d))));
  float out_x = convert_float(in_value.x);
  float out_y = convert_float(in_value.y);
  float out_z = convert_float(in_value.z);
  float out_w = convert_float(in_value.w);
  int c = d * 4;
  int index = ((b * shared_int4_0.y + y) * shared_int4_0.w + x) * shared_int4_0.x + c;

  buffer_buffer[index] = out_x;
  if (c + 1 < shared_int4_0.x) {
    buffer_buffer[index + 1] = out_y;
  }
  if (c + 2 < shared_int4_0.x) {
    buffer_buffer[index + 2] = out_z;
  }
  if (c + 3 < shared_int4_0.x) {
    buffer_buffer[index + 3] = out_w;
  }
}
)CLC";

static const int kNumPrograms = 9;
static const char* kProgramSources[kNumPrograms] = {
  kProgramSource_0,
  kProgramSource_1,
  kProgramSource_2,
  kProgramSource_3,
  kProgramSource_4,
  kProgramSource_5,
  kProgramSource_6,
  kProgramSource_7,
  kProgramSource_8,
};

static const char* kProgramOptions[kNumPrograms] = {
  "",
  "",
  "-qcom-accelerate-16-bit=true -cl-std=CL2.0",
  "",
  "",
  "",
  "-qcom-accelerate-16-bit=true -cl-std=CL2.0",
  "",
  "",
};

static const int kNumBuffers = 10;
static const size_t kBufferSizes[kNumBuffers] = {
  568197120,
  568197120,
  33423360,
  16711680,
  6144,
  6144,
  33423360,
  66846720,
  66846720,
  33423360,
};

static const cl_mem_flags kBufferFlags[kNumBuffers] = {
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x4,
  (cl_mem_flags)0x4,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
  (cl_mem_flags)0x1,
};

static const int kBufferIsSub[kNumBuffers] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};
static const int kBufferParent[kNumBuffers] = {
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
  -1,
};
static const size_t kBufferOrigin[kNumBuffers] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};

static const int kNumSvm = 1;
static const int kHasSvm = 0;
static const size_t kSvmSizes[kNumSvm] = {
  0,
};
static const cl_svm_mem_flags kSvmFlags[kNumSvm] = {
  0,
};

static const int kNumImages = 8;
static const cl_mem_object_type kImageType[kNumImages] = {
  (cl_mem_object_type)4337,
  (cl_mem_object_type)4337,
  (cl_mem_object_type)4337,
  (cl_mem_object_type)4337,
  (cl_mem_object_type)4342,
  (cl_mem_object_type)4342,
  (cl_mem_object_type)4342,
  (cl_mem_object_type)4337,
};
static const size_t kImageWidth[kNumImages] = {
  2176,
  4352,
  4352,
  2176,
  71024640,
  71024640,
  4177920,
  2176,
};
static const size_t kImageHeight[kNumImages] = {
  960,
  960,
  960,
  960,
  0,
  0,
  0,
  30,
};
static const size_t kImageDepth[kNumImages] = {
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
};
static const cl_channel_order kImageOrder[kNumImages] = {
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
  (cl_channel_order)4277,
};
static const cl_channel_type kImageTypeCode[kNumImages] = {
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
  (cl_channel_type)4317,
};
static const int kImageBackingBuffer[kNumImages] = {
  3,
  1,
  2,
  2,
  0,
  1,
  0,
  3,
};

struct SvmCopyOp {
  int dst_id;
  size_t dst_off;
  int src_id;
  size_t src_off;
  size_t size;
  const unsigned char* src_data;
};
struct SvmFillOp {
  int dst_id;
  size_t dst_off;
  size_t size;
  size_t pattern_size;
  const unsigned char* pattern;
};

static const int kNumSvmCopyOps = 0;
static const SvmCopyOp kSvmCopyOps[] = {
};

static const int kNumSvmFillOps = 0;
static const SvmFillOp kSvmFillOps[] = {
};

static const int kNumKernels = 11;
static const int kShapeH = 30;
static const int kShapeM = 4352;
static const int kShapeN = 4352;
static const int kShapeK = 128;
static const int kInputBufIds[3] = {6, 7, 8};
static const int kOutputBufId = 9;
static const int kRunOrder[] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
};


static inline bool set_kernel_args(cl_kernel * kernels, cl_mem * buffers, cl_mem * images) {
  (void)buffers;
  (void)images;
  // kernel 0 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &buffers[0]), "SetKernelArg mem 0:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[0], 1, sizeof(cl_mem), &images[1]), "SetKernelArg mem 0:1")) return false;
  static const unsigned char k0_arg2[] = {0x00, 0x11, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x40, 0x04, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[0], 2, 16, k0_arg2), "SetKernelArg val 0:2")) return false;
  static const unsigned char k0_arg3[] = {0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[0], 3, 16, k0_arg3), "SetKernelArg val 0:3")) return false;

  // kernel 1 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &buffers[1]), "SetKernelArg mem 1:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 1, sizeof(cl_mem), &images[6]), "SetKernelArg mem 1:1")) return false;
  static const unsigned char k1_arg2[] = {0x08, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x0f, 0x00, 0x20, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 2, 16, k1_arg2), "SetKernelArg val 1:2")) return false;
  static const unsigned char k1_arg3[] = {0x40, 0x04, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 3, 16, k1_arg3), "SetKernelArg val 1:3")) return false;
  static const unsigned char k1_arg4[] = {0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 4, 16, k1_arg4), "SetKernelArg val 1:4")) return false;
  static const unsigned char k1_arg5[] = {0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c};
  if (!mldrift_cl_check(clSetKernelArg(kernels[1], 5, 8, k1_arg5), "SetKernelArg val 1:5")) return false;

  // kernel 2 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &buffers[0]), "SetKernelArg mem 2:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 1, sizeof(cl_mem), &buffers[1]), "SetKernelArg mem 2:1")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 2, sizeof(cl_mem), &buffers[4]), "SetKernelArg mem 2:2")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 3, sizeof(cl_mem), &images[0]), "SetKernelArg mem 2:3")) return false;
  static const unsigned char k2_arg4[] = {0x1e, 0x00, 0x00, 0x00, 0x40, 0x04, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 4, 16, k2_arg4), "SetKernelArg val 2:4")) return false;
  static const unsigned char k2_arg5[] = {0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 5, 16, k2_arg5), "SetKernelArg val 2:5")) return false;
  static const unsigned char k2_arg6[] = {0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[2], 6, 16, k2_arg6), "SetKernelArg val 2:6")) return false;

  // kernel 3 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[3], 0, sizeof(cl_mem), &images[4]), "SetKernelArg mem 3:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[3], 1, sizeof(cl_mem), &images[7]), "SetKernelArg mem 3:1")) return false;
  static const unsigned char k3_arg2[] = {0x1e, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[3], 2, 16, k3_arg2), "SetKernelArg val 3:2")) return false;
  static const unsigned char k3_arg3[] = {0x1e, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[3], 3, 16, k3_arg3), "SetKernelArg val 3:3")) return false;

  // kernel 4 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[4], 0, sizeof(cl_mem), &buffers[1]), "SetKernelArg mem 4:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[4], 1, sizeof(cl_mem), &images[4]), "SetKernelArg mem 4:1")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[4], 2, sizeof(cl_mem), &images[7]), "SetKernelArg mem 4:2")) return false;
  static const unsigned char k4_arg3[] = {0x1e, 0x00, 0x00, 0x00, 0x40, 0x04, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[4], 3, 16, k4_arg3), "SetKernelArg val 4:3")) return false;
  static const unsigned char k4_arg4[] = {0x1e, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[4], 4, 16, k4_arg4), "SetKernelArg val 4:4")) return false;

  // kernel 5 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[5], 0, sizeof(cl_mem), &buffers[0]), "SetKernelArg mem 5:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[5], 1, sizeof(cl_mem), &images[2]), "SetKernelArg mem 5:1")) return false;
  static const unsigned char k5_arg2[] = {0x08, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x0f, 0x00, 0x40, 0x04, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[5], 2, 16, k5_arg2), "SetKernelArg val 5:2")) return false;
  static const unsigned char k5_arg3[] = {0x00, 0x11, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[5], 3, 16, k5_arg3), "SetKernelArg val 5:3")) return false;
  static const unsigned char k5_arg4[] = {0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c};
  if (!mldrift_cl_check(clSetKernelArg(kernels[5], 4, 8, k5_arg4), "SetKernelArg val 5:4")) return false;

  // kernel 6 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 0, sizeof(cl_mem), &buffers[0]), "SetKernelArg mem 6:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 1, sizeof(cl_mem), &buffers[5]), "SetKernelArg mem 6:1")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 2, sizeof(cl_mem), &images[5]), "SetKernelArg mem 6:2")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 3, sizeof(cl_mem), &images[3]), "SetKernelArg mem 6:3")) return false;
  static const unsigned char k6_arg4[] = {0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 4, 16, k6_arg4), "SetKernelArg val 6:4")) return false;
  static const unsigned char k6_arg5[] = {0x40, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 5, 16, k6_arg5), "SetKernelArg val 6:5")) return false;
  static const unsigned char k6_arg6[] = {0x00, 0xff, 0x00, 0x00, 0x40, 0x04, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 6, 16, k6_arg6), "SetKernelArg val 6:6")) return false;
  static const unsigned char k6_arg7[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[6], 7, 16, k6_arg7), "SetKernelArg val 6:7")) return false;

  // kernel 7 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[7], 0, sizeof(cl_mem), &buffers[6]), "SetKernelArg mem 7:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[7], 1, sizeof(cl_mem), &images[0]), "SetKernelArg mem 7:1")) return false;
  static const unsigned char k7_arg2[] = {0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[7], 2, 16, k7_arg2), "SetKernelArg val 7:2")) return false;

  // kernel 8 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[8], 0, sizeof(cl_mem), &buffers[7]), "SetKernelArg mem 8:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[8], 1, sizeof(cl_mem), &images[1]), "SetKernelArg mem 8:1")) return false;
  static const unsigned char k8_arg2[] = {0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[8], 2, 16, k8_arg2), "SetKernelArg val 8:2")) return false;

  // kernel 9 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[9], 0, sizeof(cl_mem), &buffers[8]), "SetKernelArg mem 9:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[9], 1, sizeof(cl_mem), &images[2]), "SetKernelArg mem 9:1")) return false;
  static const unsigned char k9_arg2[] = {0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[9], 2, 16, k9_arg2), "SetKernelArg val 9:2")) return false;

  // kernel 10 args
  if (!mldrift_cl_check(clSetKernelArg(kernels[10], 0, sizeof(cl_mem), &buffers[9]), "SetKernelArg mem 10:0")) return false;
  if (!mldrift_cl_check(clSetKernelArg(kernels[10], 1, sizeof(cl_mem), &images[3]), "SetKernelArg mem 10:1")) return false;
  static const unsigned char k10_arg2[] = {0x80, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x80, 0x08, 0x00, 0x00};
  if (!mldrift_cl_check(clSetKernelArg(kernels[10], 2, 16, k10_arg2), "SetKernelArg val 10:2")) return false;

  return true;
}

} // namespace ggml_opencl_mldrift_h30_m2176_n4352
