// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_mul_mat_Ab_Bi_8x4(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {
#if defined(USE_KAHAN_ACC) || defined(USE_F32_ACT)
#define ACCUM_FP32 1
#endif

#ifdef USE_F32_ACT
#define READ_ACT4(img, idx) read_imagef((img), (idx))
#elif defined(ACCUM_FP32)
#define READ_ACT4(img, idx) convert_float4(read_imageh((img), (idx)))
#else
#define READ_ACT4(img, idx) read_imageh((img), (idx))
#endif

#ifdef ACCUM_FP32
#define SCALE_CAST(v) ((float)(v))
#else
#define SCALE_CAST(v) (v)
#endif

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

#ifdef ACCUM_FP32
    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
#else
    half8 c0 = (half8)(0), c1 = (half8)(0), c2 = (half8)(0), c3 = (half8)(0); // 8x4 output elements
#endif
#ifdef USE_KAHAN_ACC
    float8 c0_corr = 0, c1_corr = 0, c2_corr = 0, c3_corr = 0;
#define ACC_ADD(sum, corr, val)             \
    do {                                    \
        float8 y = (val) - (corr);          \
        float8 t = (sum) + y;               \
        (corr) = (t - (sum)) - y;           \
        (sum) = t;                          \
    } while (0)
#else
#define ACC_ADD(sum, corr, val) \
    do {                        \
        (sum) += (val);         \
    } while (0)
#endif
#ifdef ACCUM_FP32
    float8 B; // registers for activations
    float4 dequantized_weights; // registers for dequantized weights
#else
    half8 B; // registers for activations
    half4 dequantized_weights; // registers for dequantized weights
#endif
    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales

    for(int i=0; i<k; i+=4){ //loop through K dimension

        B.s0123 = READ_ACT4(src1, gy*2 + (i)*(n_4));
        B.s4567 = READ_ACT4(src1, gy*2 + (i)*(n_4)+1);

        // keep (i/4) and (i/32) in parenthesis, rounds down
        // load 4 consecutive groups of 4 weights
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m)); // (i/4) because weights grouped in 4s

        // load 4 consecutive scales
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));// (i/32) because 1 scale per 32 elements

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * SCALE_CAST(scale.s0); // dequantize a row of the 16 weights
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * SCALE_CAST(scale.s1);
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * SCALE_CAST(scale.s2);
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * SCALE_CAST(scale.s3);
        ACC_ADD(c0, c0_corr, B * dequantized_weights.s0); // vector-scalar multiplication to accumulate
        ACC_ADD(c1, c1_corr, B * dequantized_weights.s1);
        ACC_ADD(c2, c2_corr, B * dequantized_weights.s2);
        ACC_ADD(c3, c3_corr, B * dequantized_weights.s3);

        // j=1
        B.s0123 = READ_ACT4(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = READ_ACT4(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * SCALE_CAST(scale.s0); // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * SCALE_CAST(scale.s1);
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * SCALE_CAST(scale.s2);
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * SCALE_CAST(scale.s3);
        ACC_ADD(c0, c0_corr, B * dequantized_weights.s0); //vector-scalar multiplication to accumulate
        ACC_ADD(c1, c1_corr, B * dequantized_weights.s1);
        ACC_ADD(c2, c2_corr, B * dequantized_weights.s2);
        ACC_ADD(c3, c3_corr, B * dequantized_weights.s3);

        // j=2
        B.s0123 = READ_ACT4(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = READ_ACT4(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * SCALE_CAST(scale.s0); // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * SCALE_CAST(scale.s1);
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * SCALE_CAST(scale.s2);
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * SCALE_CAST(scale.s3);
        ACC_ADD(c0, c0_corr, B * dequantized_weights.s0); // vector-scalar multiplication to accumulate
        ACC_ADD(c1, c1_corr, B * dequantized_weights.s1);
        ACC_ADD(c2, c2_corr, B * dequantized_weights.s2);
        ACC_ADD(c3, c3_corr, B * dequantized_weights.s3);

        // j=3
        B.s0123 = READ_ACT4(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = READ_ACT4(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * SCALE_CAST(scale.s0); // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * SCALE_CAST(scale.s1);
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * SCALE_CAST(scale.s2);
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * SCALE_CAST(scale.s3);
        ACC_ADD(c0, c0_corr, B * dequantized_weights.s0); // vector-scalar multiplication to accumulate
        ACC_ADD(c1, c1_corr, B * dequantized_weights.s1);
        ACC_ADD(c2, c2_corr, B * dequantized_weights.s2);
        ACC_ADD(c3, c3_corr, B * dequantized_weights.s3);
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements

    // conditional check if store is to a valid location. Required when N is not a multiple of 8
    // if statements allow registers to be reused for each store
    // provides a performance boost due to reduced register footprint, which increases number of concurrent waves
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }

#undef ACC_ADD
#undef SCALE_CAST
#undef READ_ACT4
#undef ACCUM_FP32
}
