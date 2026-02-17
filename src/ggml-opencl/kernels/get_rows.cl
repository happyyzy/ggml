#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}

void dequantize_q4_0_f32_soa(global uchar * q, global half * d, short il, float16 * reg) {
    global ushort * qs = (global ushort *) q;
    float d1 = il ? (*d / 16.h) : *d;
    float d2 = d1 / 256.f;
    float md = -8.h * (*d);
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int NL = 2;
    const int block_bytes = (QK4_0/2 + 2);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global uchar * src0_b = (global uchar *) src0;
    global uchar * row_base = src0_b + r*nb01 + i02*nb02 + i03*nb03;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        if (ind >= ne00) {
            return;
        }
        int block_idx = ind / NL;
        int il = ind % NL;
        global uchar * block_ptr = row_base + block_idx * block_bytes;
        global half  * d = (global half *) block_ptr;
        global uchar * q = block_ptr + 2;
        dequantize_q4_0_f32_soa(q, d, (short)il, &temp);
        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

kernel void kernel_get_rows_q4_0_soa(
        global uchar * src0_q,
        ulong offset_q,
        global half * src0_d,
        ulong offset_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0_q = (global uchar *)((global char*)src0_q + offset_q);
    src0_d = (global half  *)((global char*)src0_d + offset_d);
    src1   = (global int   *)((global char*)src1   + offset1);
    dst    = (global float *)((global char*)dst    + offsetd);

    const int NL = 2;

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int row_index = r + i11 * ne01 + i12 * ne01 * ne02;
    int blocks_per_row = ne00 / QK4_0;
    int block_base = row_index * blocks_per_row;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        if (ind >= ne00) {
            return;
        }
        int block_idx = ind / NL;
        int il = ind % NL;
        int block = block_base + block_idx;
        global uchar * q = src0_q + (block * (QK4_0/2));
        global half  * d = src0_d + block;
        dequantize_q4_0_f32_soa(q, d, (short)il, &temp);
        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

// Scalar fallback for correctness on some Adreno drivers.
kernel void kernel_get_rows_q4_0_soa_scalar(
        global uchar * src0_q,
        ulong offset_q,
        global half * src0_d,
        ulong offset_d,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0_q = (global uchar *)((global char*)src0_q + offset_q);
    src0_d = (global half  *)((global char*)src0_d + offset_d);
    src1   = (global int   *)((global char*)src1   + offset1);
    dst    = (global float *)((global char*)dst    + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int row_index = r + i11 * ne01 + i12 * ne01 * ne02;
    int blocks_per_row = ne00 / QK4_0;
    int block_base = row_index * blocks_per_row;

    for (int idx = get_local_id(0); idx < ne00; idx += get_local_size(0)) {
        int block = block_base + (idx / QK4_0);
        int in_block = idx % QK4_0;
        int j = in_block & (QK4_0/2 - 1); // 0..15

        uchar qbyte = src0_q[block * (QK4_0/2) + j];
        int qv = (in_block < QK4_0/2) ? (qbyte & 0x0F) : (qbyte >> 4);
        float d = vload_half(0, src0_d + block);
        float val = (float)(qv - 8) * d;

        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[idx] = val;
    }
}
