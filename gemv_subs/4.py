#!POPCORN leaderboard nvfp4_gemv

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

gemv_cuda_source = r"""
#include<stddef.h>
#include<cuda_fp4.h>
#include<cuda_fp16.h>

#define FP4X2_PER_16B 16
#define FP8X2_PER_16B 8
#define K_BLOCK 512
#define K_BLOCK_SMOL 32
#define ceilDiv(x, y) (((x) + (y) - 1) / (y))


template<int TILE_SIZE>
__device__ __forceinline__
void get_tile(int idx, int& tile_id, int& offset) {
    static_assert((TILE_SIZE & (TILE_SIZE - 1)) == 0, "Must be power of 2");

    constexpr int mask = TILE_SIZE - 1;
    constexpr int shift = __builtin_ctz(TILE_SIZE);

    tile_id = idx >> shift;
    offset  = idx & mask;
}

__device__ __forceinline__ void mul_and_dup_both(
    const __half2 a, const __half2 b,
    __half2 &low_dup, __half2 &high_dup)
{
    // Bit reinterpretation
    const uint32_t ra = reinterpret_cast<const uint32_t&>(a);
    const uint32_t rb = reinterpret_cast<const uint32_t&>(b);

    uint32_t prod32, lo32, hi32;

    asm volatile(
        "{\n\t"
        // Multiply two half2 registers -> f16x2
        "mul.f16x2 %0, %3, %4;\n\t"

        // Duplicate low  16-bit half: b0,b1,b0,b1
        "prmt.b32   %1, %0, %0, 0x1010;\n\t"

        // Duplicate high 16-bit half: b2,b3,b2,b3
        "prmt.b32   %2, %0, %0, 0x3232;\n\t"
        "}\n"
        : "=&r"(prod32), "=&r"(lo32), "=&r"(hi32)
        : "r"(ra), "r"(rb)
    );

    low_dup  = reinterpret_cast<__half2&>(lo32);
    high_dup = reinterpret_cast<__half2&>(hi32);
}


__device__ __forceinline__ void dup_both(
    const __half2 a,
    __half2 &low_dup, __half2 &high_dup)
{
    // Bit reinterpretation
    const uint32_t ra = reinterpret_cast<const uint32_t&>(a);

    uint32_t lo32, hi32;

    asm volatile(
        "{\n\t"
        // Duplicate low  16-bit half: b0,b1,b0,b1
        "prmt.b32   %0, %2, %2, 0x1010;\n\t"
        // Duplicate high 16-bit half: b2,b3,b2,b3
        "prmt.b32   %1, %2, %2, 0x3232;\n\t"
        "}\n"
        : "=&r"(lo32), "=&r"(hi32)
        : "r"(ra)
    );

    low_dup  = reinterpret_cast<__half2&>(lo32);
    high_dup = reinterpret_cast<__half2&>(hi32);
}


__device__ __forceinline__ float fma_reduce_fp16x2(
    const uint32_t* a_reg_half2,  // Pointer to 16 half2 values (a[0:15])
    const uint32_t* b_reg_half2,  // Pointer to 16 half2 values (b[0:15])
    uint32_t scale0_h_u32,        // First scale factor as half2 in uint32_t
    uint32_t scale1_h_u32         // Second scale factor as half2 in uint32_t
) {
    uint32_t acc_h0_u32 = 0;
    uint32_t acc_h1_u32 = 0;
    float final_accum = 0.0f;
    float tmp_x, tmp_y;

    asm volatile(
        "{\n\t"
        // Interleave acc_h0 and acc_h1 FMAs
        ".reg .b16 h0, h1;\n\t"
        //".reg .v2 .b16 hpair;\n\t"
        ".reg .f32 f0, f1;\n\t"
        ".reg .b32 rtmp, acc0, acc1, acc2, acc3;\n\t"
        "mov.b32 acc0, 0;\n\t"
        "mov.b32 acc1, 0;\n\t"
        "mov.b32 acc2, 0;\n\t"
        "mov.b32 acc3, 0;\n\t"

        "fma.rn.f16x2 acc0, %5, %21, acc0;\n"    // acc_h0[0]: a[0] * b[0]
        "fma.rn.f16x2 acc2, %13, %29, acc2;\n"   // acc_h1[0]: a[8] * b[8]
        "fma.rn.f16x2 acc1, %6, %22, acc1;\n"    // acc_h0[1]: a[1] * b[1]
        "fma.rn.f16x2 acc3, %14, %30, acc3;\n"   // acc_h1[1]: a[9] * b[9]
        "fma.rn.f16x2 acc0, %7, %23, acc0;\n"    // acc_h0[2]: a[2] * b[2]
        "fma.rn.f16x2 acc2, %15, %31, acc2;\n"   // acc_h1[2]: a[10] * b[10]
        "fma.rn.f16x2 acc1, %8, %24, acc1;\n"    // acc_h0[3]: a[3] * b[3]
        "fma.rn.f16x2 acc3, %16, %32, acc3;\n"   // acc_h1[3]: a[11] * b[11]
        "fma.rn.f16x2 acc0, %9, %25, acc0;\n"    // acc_h0[4]: a[4] * b[4]
        "fma.rn.f16x2 acc2, %17, %33, acc2;\n"   // acc_h1[4]: a[12] * b[12]
        "fma.rn.f16x2 acc1, %10, %26, acc1;\n"   // acc_h0[5]: a[5] * b[5]
        "fma.rn.f16x2 acc3, %18, %34, acc3;\n"   // acc_h1[5]: a[13] * b[13]
        "fma.rn.f16x2 acc0, %11, %27, acc0;\n"   // acc_h0[6]: a[6] * b[6]
        "fma.rn.f16x2 acc2, %19, %35, acc2;\n"   // acc_h1[6]: a[14] * b[14]
        "fma.rn.f16x2 acc1, %12, %28, acc1;\n"   // acc_h0[7]: a[7] * b[7]
        "fma.rn.f16x2 acc3, %20, %36, acc3;\n"   // acc_h1[7]: a[15] * b[15]
        // Combine each group of 8 into one accumulator
        "add.rn.f16x2 acc0, acc1, acc0;\n\t"
        "add.rn.f16x2 acc2, acc3, acc2;\n\t"

        // Apply scales: racc0 = scale0 * sum0 + scale1 * sum1
        "mul.rn.f16x2 acc0, acc0,  %37;\n\t"
        "fma.rn.f16x2 acc0, acc2,  %38, acc0;\n\t"

        // Scale and combine
        //"mul.rn.f16x2 %0, %0, %37;\n"        // acc_h0 *= scale0
        //"fma.rn.f16x2 %0, %1, %38, %0;\n"    // acc_h0 += acc_h1 * scale1
        // Convert to float and extract components
        //
        "mov.b32 {h0, h1}, acc0;\n"  // low 16 bits
        "cvt.f32.f16 f0, h0;\n"
        "cvt.f32.f16 f1, h1;\n"
        "add.f32 %4, f0, f1;\n"
        "}\n"
        : "+r"(acc_h0_u32), "+r"(acc_h1_u32),  // %0, %1
          "=f"(tmp_x), "=f"(tmp_y),             // %2, %3
          "+f"(final_accum)                     // %4
        : "r"(a_reg_half2[0]), "r"(a_reg_half2[1]),     // %5-6
          "r"(a_reg_half2[2]), "r"(a_reg_half2[3]),     // %7-8
          "r"(a_reg_half2[4]), "r"(a_reg_half2[5]),     // %9-10
          "r"(a_reg_half2[6]), "r"(a_reg_half2[7]),     // %11-12
          "r"(a_reg_half2[8]), "r"(a_reg_half2[9]),     // %13-14
          "r"(a_reg_half2[10]), "r"(a_reg_half2[11]),   // %15-16
          "r"(a_reg_half2[12]), "r"(a_reg_half2[13]),   // %17-18
          "r"(a_reg_half2[14]), "r"(a_reg_half2[15]),   // %19-20
          "r"(b_reg_half2[0]), "r"(b_reg_half2[1]),     // %21-22
          "r"(b_reg_half2[2]), "r"(b_reg_half2[3]),     // %23-24
          "r"(b_reg_half2[4]), "r"(b_reg_half2[5]),     // %25-26
          "r"(b_reg_half2[6]), "r"(b_reg_half2[7]),     // %27-28
          "r"(b_reg_half2[8]), "r"(b_reg_half2[9]),     // %29-30
          "r"(b_reg_half2[10]), "r"(b_reg_half2[11]),   // %31-32
          "r"(b_reg_half2[12]), "r"(b_reg_half2[13]),   // %33-34
          "r"(b_reg_half2[14]), "r"(b_reg_half2[15]),   // %35-36
          "r"(scale0_h_u32), "r"(scale1_h_u32)          // %37-38
    );

    return final_accum;
}


// Helper to convert __half2 to uint32_t if needed:
__device__ __forceinline__ uint32_t half2_to_u32(__half2 h) {
    return *reinterpret_cast<uint32_t*>(&h);
}


__device__ __forceinline__
__half2 load_and_convert_fp8x2_pred(const uint16_t* ptr, bool pred) {
    uint32_t out;
    asm volatile(
        "{\n"
        "  .reg .b16 raw;\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;          \n"
        "  @p ld.global.cs.u16 raw, [%1];\n"  // or .ca if you prefer
        "  @!p mov.b16 raw, 0x3c3c;       \n"  // FP8 E4M3 "1,1" (example)
        "  cvt.rn.f16x2.e4m3x2 %0, raw;  \n"
        "}\n"
        : "=r"(out)
        : "l"(ptr), "r"((int)pred)
    );
    return *reinterpret_cast<__half2*>(&out);
}


__device__ __forceinline__
__half2 fp4x2_e2m1_to_half2_ptx(uint16_t raw_bits) {
    uint32_t out_bits;

    asm volatile(
        "{\n"
        "  .reg .b8  b;\n"
        "  .reg .b32 tmp;\n"

        // take low 8 bits = packed fp4x2
        "  mov.b8 b, %1;\n"

        // convert fp4x2 -> f16x2
        "  cvt.rn.f16x2.e2m1x2 %0, b;\n"
        "}\n"
        : "=r"(out_bits)
        : "h"(raw_bits)
    );

    return *reinterpret_cast<__half2*>(&out_bits);
}

__device__ __forceinline__
void convert16_fp4x2_to_half2(
    const uint32_t (&in)[4],    // 4× packed FP4x2 words
    uint32_t (&out)[16]         // 16× half2 bit patterns
) {
    //asm (
    asm volatile(
        "{\n"
        "  .reg .b8 b0, b1, b2, b3;\n"

        // in[0] -> out[0..3]
        "  mov.b32 {b0, b1, b2, b3}, %16;\n"
        "  cvt.rn.f16x2.e2m1x2 %0,  b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %1,  b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %2,  b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %3,  b3;\n"

        // in[1] -> out[4..7]
        "  mov.b32 {b0, b1, b2, b3}, %17;\n"
        "  cvt.rn.f16x2.e2m1x2 %4,  b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %5,  b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %6,  b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %7,  b3;\n"

        // in[2] -> out[8..11]
        "  mov.b32 {b0, b1, b2, b3}, %18;\n"
        "  cvt.rn.f16x2.e2m1x2 %8,  b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %9,  b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %10, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %11, b3;\n"

        // in[3] -> out[12..15]
        "  mov.b32 {b0, b1, b2, b3}, %19;\n"
        "  cvt.rn.f16x2.e2m1x2 %12, b0;\n"
        "  cvt.rn.f16x2.e2m1x2 %13, b1;\n"
        "  cvt.rn.f16x2.e2m1x2 %14, b2;\n"
        "  cvt.rn.f16x2.e2m1x2 %15, b3;\n"
        "}\n"
        : // 16 outputs: 16 half2 bit patterns
          "=r"(out[0]),  "=r"(out[1]),  "=r"(out[2]),  "=r"(out[3]),
          "=r"(out[4]),  "=r"(out[5]),  "=r"(out[6]),  "=r"(out[7]),
          "=r"(out[8]),  "=r"(out[9]),  "=r"(out[10]), "=r"(out[11]),
          "=r"(out[12]), "=r"(out[13]), "=r"(out[14]), "=r"(out[15])
        : // 4 packed FP4x2 inputs
          "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3])
    );
}

__device__ __forceinline__ 
uint4 pred_ld_uint4_cs(const uint4* ptr, bool pred) {
    uint4 v;
    v.x = v.y = v.z = v.w = 0u;
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ne.s32 p, %4, 0;             // p = (pred != 0)\n\t"
        "  @p ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%5];\n\t"
        "}\n"
        : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w)
        : "r"((int)pred), "l"(ptr)
    );
    return v;
}

__device__ __forceinline__ 
uint4 pred_ld_uint4_ca(const uint4* ptr, bool pred) {
    uint4 v;
    v.x = v.y = v.z = v.w = 0u;
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  setp.ne.s32 p, %4, 0;             // p = (pred != 0)\n\t"
        "  @p ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%5];\n\t"
        "}\n"
        : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w)
        : "r"((int)pred), "l"(ptr)
    );
    return v;
}

__device__ __forceinline__
__half2 fp4x2_e2m1_to_half2_ptx(__nv_fp4x2_e2m1 v) {
    uint16_t raw_bits = *reinterpret_cast<uint16_t*>(&v);
    return fp4x2_e2m1_to_half2_ptx(raw_bits);
}

__device__ __forceinline__ __half2 fp4x2_e2m1_to_half2(__nv_fp4x2_e2m1 v) {
    __nv_fp4x2_storage_t raw = v.__x;  // packed 2×fp4
    __half2_raw hraw = __nv_cvt_fp4x2_to_halfraw2(raw, __NV_E2M1);
    return *reinterpret_cast<__half2*>(&hraw);
}

__device__ __forceinline__ __half2 fp8x2_e4m3_to_half2(__nv_fp8x2_e4m3 v) {
    __nv_fp8x2_storage_t raw = v.__x;
    __half2_raw hraw = __nv_cvt_fp8x2_to_halfraw2(raw, __NV_E4M3);
    return *reinterpret_cast<__half2*>(&hraw);
}

__device__ __forceinline__ __half fp8_e4m3_to_half(__nv_fp8_e4m3 v) {
    __nv_fp8_storage_t raw = v.__x;
    __half_raw hraw = __nv_cvt_fp8_to_halfraw(raw, __NV_E4M3);
    return *reinterpret_cast<__half*>(&hraw);
}


template<int M, int K, int M_BLOCK>
__launch_bounds__(M_BLOCK*32)
__global__ void gemv_kernel(
		const __nv_fp4x2_e2m1* A, 
		const __nv_fp4x2_e2m1* B, 
    const __nv_fp8x2_e4m3* SFA,
    const __nv_fp8x2_e4m3* SFB,
		half* C
		) {
  int threadID = threadIdx.x;
  int warpID, laneID; 
  get_tile<32>(threadID, warpID, laneID);
  int rowID = warpID;
  static_assert(sizeof(__nv_fp4x2_e2m1) == 1, "fp4x2 is not 1 byte");
  static_assert(sizeof(uint4) == 16, "uint4 not 16 bytes");


  constexpr int MK = M * K;
  constexpr int N = 128;
  constexpr int NK = N * K;
  constexpr int MK_SF = MK / 16;
  constexpr int NK_SF = NK / 16;
  constexpr int K_SF = K / 16;

  int blockRowIdx = blockIdx.x * M_BLOCK;
  int threadRowIdx = blockRowIdx + rowID;
  int batchBlockIdx = blockIdx.z;

  if (threadRowIdx >= M)
    return;

  int batchOffset = MK * batchBlockIdx;
  int bBatchOffset = NK * batchBlockIdx;
  int rowOffset =  K * threadRowIdx;
  int cOffset = (M * batchBlockIdx + blockRowIdx);

  // scale factor offsets
  // Have K//16 fp8 values per row 
  // We are interpreting the pointer as fp8x2 so we have K//32 values per row
  int sfaBatchOffset = MK_SF * batchBlockIdx;
  int sfbBatchOffset = NK_SF * batchBlockIdx;
  int sfaRowOffset = K_SF * threadRowIdx;

  int laneOffset = laneID * FP4X2_PER_16B;
  const __nv_fp4x2_e2m1 *gALanePtr = A + batchOffset + rowOffset + laneOffset;
  const __nv_fp4x2_e2m1 *gBLanePtr = B + bBatchOffset + laneOffset;
  const uint16_t *gSFALanePtr = reinterpret_cast<const uint16_t *>(SFA + sfaBatchOffset + sfaRowOffset + laneID);
  const uint16_t *gSFBLanePtr = reinterpret_cast<const uint16_t *>(SFB + sfbBatchOffset + laneID);


  constexpr int NUM_TILES = (K + K_BLOCK - 1) / K_BLOCK;
  constexpr int K_STAGES = 4;
  constexpr int K_STAGE_MASK = K_STAGES - 1;
  constexpr int PRELOAD_K = K_STAGES * K_BLOCK;
  constexpr int PRELOAD_K_SMOL = K_STAGES * K_BLOCK_SMOL;
  uint32_t  a_reg_fp4x2[K_STAGES][4];
  uint32_t  b_reg_fp4x2[K_STAGES][4];

  __nv_fp8x2_e4m3 sfa_reg_fp8x2[K_STAGES];
  __nv_fp8x2_e4m3 sfb_reg_fp8x2[K_STAGES];
  //__half2 a_reg_half2[K_STAGES][16];
  //__half2 b_reg_half2[K_STAGES][16];
  uint32_t a_reg_half2[K_STAGES][16];
  uint32_t b_reg_half2[K_STAGES][16];
  __half2 sfa_vals_h[K_STAGES]; 
  __half2 sfb_vals_h[K_STAGES];
  float final_accum = 0.0f;
  constexpr uint16_t FP8_E4M3_ONE2 = 0x3838;
  constexpr uint4  UINT4_ZERO = uint4{0,0,0,0};
  const __half2 HALF2_ZERO = __float2half2_rn(0.0f);

  //const uint4 *gA_ptr, *gB_ptr;
  const __nv_fp4x2_e2m1 *gA_ptr, *gB_ptr;
  const uint16_t *gSFA_ptr, *gSFB_ptr;
  // init pointers
  //gA_ptr   = reinterpret_cast<const uint4*>(gALanePtr);
  //gB_ptr   = reinterpret_cast<const uint4*>(gBLanePtr);
  gA_ptr   = gALanePtr;
  gB_ptr   = gBLanePtr;
  gSFA_ptr  = gSFALanePtr;
  gSFA_ptr  = gSFALanePtr;
  gSFB_ptr  = gSFBLanePtr;
  
  // Warm up pipeline: prefetch up to K_STAGES tiles
  bool in_range;
  int k_idx = laneOffset;
  #pragma unroll
  for (int stage=0; stage<K_STAGES; ++stage) {
    in_range = k_idx < K;
    *(reinterpret_cast<uint4 *>(&a_reg_fp4x2[stage][0])) = pred_ld_uint4_cs(reinterpret_cast<const uint4 *>(gA_ptr), in_range); 
    *(reinterpret_cast<uint4 *>(&b_reg_fp4x2[stage][0])) = pred_ld_uint4_ca(reinterpret_cast<const uint4 *>(gB_ptr), in_range);
    *(reinterpret_cast<uint16_t *>(&sfa_reg_fp8x2[stage])) = in_range ? __ldcs(gSFA_ptr) : FP8_E4M3_ONE2;
    *(reinterpret_cast<uint16_t *>(&sfb_reg_fp8x2[stage])) = in_range ? __ldca(gSFB_ptr) : FP8_E4M3_ONE2;
    gA_ptr += K_BLOCK;
    gB_ptr += K_BLOCK;
    gSFA_ptr += 32;
    gSFB_ptr += 32;
    k_idx += K_BLOCK;
  }
  // Reset all pointers to what they shold be here (this should be not needed)
  //gA_ptr   = reinterpret_cast<const uint4*>(gALanePtr + PRELOAD_K);
  //gB_ptr   = reinterpret_cast<const uint4*>(gBLanePtr + PRELOAD_K);
  //gSFA_ptr  = gSFALanePtr + PRELOAD_K_SMOL;
  //gSFB_ptr  = gSFBLanePtr + PRELOAD_K_SMOL;
  //k_idx = laneOffset + PRELOAD_K;
  int stage = 0;
  __half2 acc = HALF2_ZERO;
  __half2 scale0_h, scale1_h; 
  for (int compute_tile=0;compute_tile<NUM_TILES; ++compute_tile) {
    stage = compute_tile & K_STAGE_MASK;
    // DO THE COMPUTE
    sfa_vals_h[stage] = (fp8x2_e4m3_to_half2(sfa_reg_fp8x2[stage]));
    sfb_vals_h[stage] = (fp8x2_e4m3_to_half2(sfb_reg_fp8x2[stage]));
    convert16_fp4x2_to_half2(a_reg_fp4x2[stage], a_reg_half2[stage]);
    __half2 scale = __hmul2(sfa_vals_h[stage], sfb_vals_h[stage]);
    //mul_and_dup_both(sfa_vals_h[stage], sfb_vals_h[stage], scale0_h, scale1_h);
    convert16_fp4x2_to_half2(b_reg_fp4x2[stage], b_reg_half2[stage]);
    scale0_h = __half2half2(__low2half(scale));
    scale1_h = __half2half2(__high2half(scale));
    //dup_both(scale, scale0_h, scale1_h);
    __half2 acc_h0 = HALF2_ZERO;
    __half2 acc_h1 = HALF2_ZERO;
    __half2* a_ptr = reinterpret_cast<__half2*>(&a_reg_half2[stage][0]);
    __half2* b_ptr = reinterpret_cast<__half2*>(&b_reg_half2[stage][0]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc_h0 = __hfma2(a_ptr[i],    b_ptr[i],    acc_h0);
      acc_h1 = __hfma2(a_ptr[i+8],  b_ptr[i+8],  acc_h1);
    }
    //acc_h0 = __hmul2(acc_h0, scale0_h);
    acc = __hfma2(acc_h0, scale0_h, acc);
    acc = __hfma2(acc_h1, scale1_h, acc);
    //acc = __hadd2(acc, acc_h0);
    //float2 tmp = __half22float2(acc_h0);
    //final_accum = final_accum + tmp.x + tmp.y;

    //uint32_t scale0_u32 = half2_to_u32(scale0_h);
    //uint32_t scale1_u32 = half2_to_u32(scale1_h);
    //final_accum += fma_reduce_fp16x2(a_reg_half2[stage], b_reg_half2[stage], scale0_u32, scale1_u32);
    // then load next tile into same slot
    in_range = k_idx < K;
    //*(reinterpret_cast<uint4 *>(&a_reg_fp4x2[stage][0])) = pred_ld_uint4_cs(gA_ptr, in_range); 
    //*(reinterpret_cast<uint4 *>(&b_reg_fp4x2[stage][0])) = pred_ld_uint4_ca(gB_ptr, in_range);
    *(reinterpret_cast<uint4 *>(&a_reg_fp4x2[stage][0])) = pred_ld_uint4_cs(reinterpret_cast<const uint4 *>(gA_ptr), in_range);
    *(reinterpret_cast<uint4 *>(&b_reg_fp4x2[stage][0])) = pred_ld_uint4_ca(reinterpret_cast<const uint4 *>(gB_ptr), in_range);
    *(reinterpret_cast<uint16_t *>(&sfa_reg_fp8x2[stage])) = in_range ? __ldcs(gSFA_ptr) : FP8_E4M3_ONE2;
    *(reinterpret_cast<uint16_t *>(&sfb_reg_fp8x2[stage])) = in_range ? __ldca(gSFB_ptr) : FP8_E4M3_ONE2;
    // advance da pointers
    gA_ptr += K_BLOCK;
    gB_ptr += K_BLOCK;
    gSFA_ptr += 32;
    gSFB_ptr += 32;
    k_idx += K_BLOCK;
  }
  // at this point each thread contains the sum of it's strided values in the row
  // need to use a warp reduction on each warp to compute final row sum
  float2 tmp = __half22float2(acc);
  final_accum = tmp.x + tmp.y;
  constexpr unsigned FULL_MASK = 0xffffffff;
  for (int offset = 16; offset > 0; offset >>= 1) {
    final_accum += __shfl_down_sync(FULL_MASK, final_accum, offset);
  }
  if (laneID == 0) {
    C[cOffset + rowID] = __float2half(final_accum);
  }
}


template<int M, int K, int M_BLOCK=4>
void launch_gemv(
const __nv_fp4x2_e2m1* A,
const __nv_fp4x2_e2m1* B,
const __nv_fp8x2_e4m3* SFA,
const __nv_fp8x2_e4m3* SFB,
half* C,
int L)
{
    int threads = M_BLOCK * 32;
    dim3 grid(ceilDiv(M, M_BLOCK), 1, L);
    auto dis_kernel = gemv_kernel<M, K, M_BLOCK>;
    /*
    cudaFuncSetAttribute(
        dis_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);
        */
    //gemv_kernel<M, K><<<grid, threads>>>(A, B, SFA, SFB, C);
    dis_kernel<<<grid, threads>>>(A, B, SFA, SFB, C);
}


torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor SFA, torch::Tensor SFB, torch::Tensor C) {
    //TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
    //TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
    //TORCH_CHECK(SFA.device().is_cuda(), "Tensor SFA must be a CUDA tensor");
    //TORCH_CHECK(SFB.device().is_cuda(), "Tensor SFB must be a CUDA tensor");
    //TORCH_CHECK(C.device().is_cuda(), "Tensor C must be a CUDA tensor");
    
    int M = A.size(0); 
    int K = A.size(1); 
    int L = A.size(2); 

    //dim3 block(M_BLOCK * 32, 1, 1);
    //printf("Problem size M: %d, K: %d, N: %d, L: %d \n", M, K, N, L);
    //printf("Threads per block: %d, Block dims (%d, 1, %d)\n", threads, grid.x, grid.z);

    auto A_ptr = reinterpret_cast<__nv_fp4x2_e2m1*>(A.data_ptr());
    auto B_ptr = reinterpret_cast<__nv_fp4x2_e2m1*>(B.data_ptr());
    auto SFA_ptr = reinterpret_cast<__nv_fp8x2_e4m3*>(SFA.data_ptr());
    auto SFB_ptr = reinterpret_cast<__nv_fp8x2_e4m3*>(SFB.data_ptr());
    auto C_ptr = reinterpret_cast<__half*>(C.data_ptr());
  
    // set max l1 for benchmarks
    /*
    if (M==7168 && K==8192) {
      auto dis_kernel = gemv_kernel<7168, 8192>;
      cudaFuncSetAttribute(
          dis_kernel,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          //cudaSharedmemCarveoutMaxL1
          cudaSharedmemCarveoutMaxShared
          );
    }
    else if (M==4096 && K==3584) {
      auto dis_kernel = gemv_kernel<4096, 3384>;
      cudaFuncSetAttribute(
          dis_kernel,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          //cudaSharedmemCarveoutMaxL1);
          ////cudaSharedmemCarveoutMaxL1
          cudaSharedmemCarveoutMaxShared
          );
    }
    else if (M==7168 && K==1024) {
      auto dis_kernel = gemv_kernel<7168, 1024>;
      cudaFuncSetAttribute(
          dis_kernel,
          cudaFuncAttributePreferredSharedMemoryCarveout,
          //cudaSharedmemCarveoutMaxL1);
          //cudaSharedmemCarveoutMaxL1
          cudaSharedmemCarveoutMaxShared
          );
    }
    */
    
    if (M==128 && K==128) {
      launch_gemv<128, 128>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==128 && K==768) {
      launch_gemv<128, 768>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==128 && K==1536) {
      launch_gemv<128, 1536>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==256 && K==3584) {
      launch_gemv<256, 3584>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==2432 && K==2304) {
      launch_gemv<2432, 2304>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==384 && K==3584) {
      launch_gemv<384, 3584>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==512 && K==256) {
      launch_gemv<512, 256>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==512 && K==2048) {
      launch_gemv<512, 2048>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==512 && K==768) {
      launch_gemv<512, 768>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==7168 && K==8192) {
      launch_gemv<7168, 8192, 1>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==4096 && K==3584) {
      launch_gemv<4096, 3584, 3>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else if (M==7168 && K==1024) {
      launch_gemv<7168, 1024, 4>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L);
    }
    else {
        throw std::runtime_error("Unsupported (M, K) combination");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}
"""

gemv_cpp_source = """
#include <torch/extension.h>

torch::Tensor gemv_cuda(
  torch::Tensor A, 
  torch::Tensor B, 
  torch::Tensor SFA, 
  torch::Tensor SFB, 
  torch::Tensor C);
"""
extra_cuda_cflags = [
    "-O3",
    "--use_fast_math",
    "--fmad=true",
    "--ftz=true",
    #"-Xcompiler", "-fno-strict-aliasing",

    # Aggressive math optimizations
    "-Xptxas=-O3",

    # Cache behavior
    #"-Xptxas=-dlcm=ca",

    # For debugging performance
    "-Xptxas=--warn-on-spills",
    "-Xptxas=-v",

    # Blackwell target
    #"--gpu-architecture=sm_100a",
    "-gencode=arch=compute_100a,code=sm_100a",
]

extra_cflags = [
    "-O3",
    "-ffast-math",
    "-fno-strict-aliasing",
]


gemv_module = load_inline(
    name='gemv_cuda',
    cpp_sources=gemv_cpp_source,
    cuda_sources=gemv_cuda_source,
    functions=['gemv_cuda'],
    verbose=True,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_cflags=extra_cflags,
)



def gemv_cuda(A, B, SFA, SFB, C):
    if not A.is_cuda or not B.is_cuda or not SFA.is_cuda or not SFB.is_cuda or not C.is_cuda:
        raise RuntimeError("Both tensors must be on GPU")
    return gemv_module.gemv_cuda(A, B, SFA, SFB, C)


# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b


def custom_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """
    a_ref, b_ref, sfa, sfb, _, _, c_ref = data
    m, k, l = a_ref.shape
    n, k, l = b_ref.shape
    """
    print(f"K is {k}, n is {n}")
    print(f"A shape {a_ref.shape}")
    print(f"A shape {a_ref.stride()}")
    print(f"SFA shape {sfa.shape}")
    print(f"SFA shape {sfa.stride()}")
    print(f"B shape {b_ref.shape}")
    print(f"B shape {b_ref.stride()}")
    print(f"SFB shape {sfb.shape}")
    print(f"SFB shape {sfb.stride()}")
    print(f"C shape {c_ref.shape}")
    print(f"C shape {c_ref.stride()}")
    """

    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape
    #print(sfa.shape, sfa.stride())
    #print(f"SFA[0,0:32,0]: {sfa[0,:32,0].reshape(-1,2)}")
    gemv_cuda(a_ref, b_ref, sfa, sfb, c_ref)
    #torch.cuda.synchronize()
    #print(c_ref)
    return c_ref
