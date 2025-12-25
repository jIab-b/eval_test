#!POPCORN leaderboard nvfp4_gemv

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


# CUDA SOURCE CODE

cuda_source = """
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda/ptx>
#include<cuda_awbarrier_primitives.h>

namespace ptx = cuda::ptx;


#define FULL_MASK 0xffffffff

__inline__ __device__ void multiply_and_accumulate(
    int4 a_packed,
    int4 b_packed,
    __nv_fp8x2_storage_t sfa_fp8x2,
    __nv_fp8x2_storage_t sfb_fp8x2,
    int* result_0,
    int* result_1,
    int* result_2,
    int* result_3
) {
    asm volatile( \\
        "{\\n" \\
        // declare registers for A / B tensors
        ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\\n" \\
        ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\\n" \\
        ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\\n" \\
        ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\\n" \\
        ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\\n" \\
        ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\\n" \\
        ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\\n" \\
        ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\\n" \\

        // declare registers for accumulators
        ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\\n" \\
        ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\\n" \\
        ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\\n" \\
        ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\\n" \\

        // declare registers for scaling factors
        ".reg .f16x2 sfa_f16x2;\\n" \\
        ".reg .f16x2 sfb_f16x2;\\n" \\
        ".reg .f16x2 sf_f16x2;\\n" \\
        
        // declare registers for conversion
        ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n" \\
        ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\\n" \\
        ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n" \\
        ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\\n" \\
        ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\\n" \\
        ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\\n" \\
        ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\\n" \\
        ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\\n" \\
        ".reg .f16 result_f16, lane0, lane1;\\n" \\
        ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\\n" \\

        // convert scaling factors from fp8 to f16x2
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %4;\\n" \\
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %5;\\n" \\
        
        // clear accumulators
        "mov.b32 accum_0_0, 0;\\n" \\
        "mov.b32 accum_0_1, 0;\\n" \\
        "mov.b32 accum_0_2, 0;\\n" \\
        "mov.b32 accum_0_3, 0;\\n" \\
        "mov.b32 accum_1_0, 0;\\n" \\
        "mov.b32 accum_1_1, 0;\\n" \\
        "mov.b32 accum_1_2, 0;\\n" \\
        "mov.b32 accum_1_3, 0;\\n" \\
        "mov.b32 accum_2_0, 0;\\n" \\
        "mov.b32 accum_2_1, 0;\\n" \\
        "mov.b32 accum_2_2, 0;\\n" \\
        "mov.b32 accum_2_3, 0;\\n" \\
        "mov.b32 accum_3_0, 0;\\n" \\
        "mov.b32 accum_3_1, 0;\\n" \\
        "mov.b32 accum_3_2, 0;\\n" \\
        "mov.b32 accum_3_3, 0;\\n" \\
        
        // multiply, unpacking and permuting scale factors
        "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\\n" \\
        "mov.b32 {lane0, lane1}, sf_f16x2;\\n" \\
        "mov.b32 mul_f16x2_0, {lane0, lane0};\\n" \\
        "mov.b32 mul_f16x2_1, {lane1, lane1};\\n" \\

        // unpacking A and B tensors
        "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %6;\\n" \\
        "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, %7;\\n" \\
        "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %8;\\n" \\
        "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, %9;\\n" \\
        "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %10;\\n" \\
        "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, %11;\\n" \\
        "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %12;\\n" \\
        "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, %13;\\n" \\

        // convert A and B tensors from fp4 to f16x2

        // A[0 - 7] and B[0 - 7]
        "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\\n" \\

        // A[8 - 15] and B[8 - 15]
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\\n" \\

        // A[16 - 23] and B[16 - 23]
        "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\\n" \\

        // A[24 - 31] and B[24 - 31]
        "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\\n" \\
        "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\\n" \\

        // fma for A[0 - 7] and B[0 - 7]
        "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_0_4, accum_0_0;\\n" \\
        "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_0_5, accum_0_1;\\n" \\
        "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_0_6, accum_0_2;\\n" \\
        "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_0_7, accum_0_3;\\n" \\

        // fma for A[8 - 15] and B[8 - 15]
        "fma.rn.f16x2 accum_1_0, cvt_1_0, cvt_1_4, accum_1_0;\\n" \\
        "fma.rn.f16x2 accum_1_1, cvt_1_1, cvt_1_5, accum_1_1;\\n" \\
        "fma.rn.f16x2 accum_1_2, cvt_1_2, cvt_1_6, accum_1_2;\\n" \\
        "fma.rn.f16x2 accum_1_3, cvt_1_3, cvt_1_7, accum_1_3;\\n" \\

        // fma for A[16 - 23] and B[16 - 23]
        "fma.rn.f16x2 accum_2_0, cvt_2_0, cvt_2_4, accum_2_0;\\n" \\
        "fma.rn.f16x2 accum_2_1, cvt_2_1, cvt_2_5, accum_2_1;\\n" \\
        "fma.rn.f16x2 accum_2_2, cvt_2_2, cvt_2_6, accum_2_2;\\n" \\
        "fma.rn.f16x2 accum_2_3, cvt_2_3, cvt_2_7, accum_2_3;\\n" \\

        // fma for A[24 - 31] and B[24 - 31]
        "fma.rn.f16x2 accum_3_0, cvt_3_0, cvt_3_4, accum_3_0;\\n" \\
        "fma.rn.f16x2 accum_3_1, cvt_3_1, cvt_3_5, accum_3_1;\\n" \\
        "fma.rn.f16x2 accum_3_2, cvt_3_2, cvt_3_6, accum_3_2;\\n" \\
        "fma.rn.f16x2 accum_3_3, cvt_3_3, cvt_3_7, accum_3_3;\\n" \\

        // tree reduction for accumulators
        "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\\n" \\
        "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\\n" \\
        "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\\n" \\
        "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\\n" \\
        "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\\n" \\
        "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\\n" \\
        "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\\n" \\
        "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\\n" \\

        "fma.rn.f16x2 %0, accum_0_0, mul_f16x2_0, %0;\\n" \\
        "fma.rn.f16x2 %1, accum_0_2, mul_f16x2_0, %1;\\n" \\
        "fma.rn.f16x2 %2, accum_1_0, mul_f16x2_0, %2;\\n" \\
        "fma.rn.f16x2 %3, accum_1_2, mul_f16x2_0, %3;\\n" \\
        

        "fma.rn.f16x2 %0, accum_2_0, mul_f16x2_1, %0;\\n" \\
        "fma.rn.f16x2 %1, accum_2_2, mul_f16x2_1, %1;\\n" \\
        "fma.rn.f16x2 %2, accum_3_0, mul_f16x2_1, %2;\\n" \\
        "fma.rn.f16x2 %3, accum_3_2, mul_f16x2_1, %3;\\n" \\

        "}\\n"
        : "+r"(*result_0), "+r"(*result_1), "+r"(*result_2), "+r"(*result_3)    // 0, 1, 2, 3
        : "h"(sfa_fp8x2), "h"(sfb_fp8x2),                   // 4, 5
            "r"(a_packed.x), "r"(b_packed.x),               // 6, 7
            "r"(a_packed.y), "r"(b_packed.y),               // 8, 9
            "r"(a_packed.z), "r"(b_packed.z),               // 10, 11
            "r"(a_packed.w), "r"(b_packed.w)                // 12, 13
    );
}


__global__ void gemv_kernel_4096_7168(
    const __nv_fp4x2_storage_t* __restrict__ a,
    const __nv_fp4x2_storage_t* __restrict__ b,
    const __nv_fp8_e4m3* __restrict__ sfa,
    const __nv_fp8_e4m3* __restrict__ sfb,
    __half* __restrict__ c
) {
    const int M = 4096;
    const int K = 7168;

    extern __shared__ unsigned char shared_storage[];
    auto* b_shared = reinterpret_cast<__nv_fp4x2_storage_t*>(shared_storage);
    auto* sfb_shared = reinterpret_cast<__nv_fp8_e4m3*>(b_shared + (K / 2));
    __shared__ __half c_shared[32];

    b += blockIdx.y * (K / 2) * 128;
    sfb += blockIdx.y * (K / 16) * 128;

    for (int i = threadIdx.y * 32 + threadIdx.x; i < K / 32; i += blockDim.y * blockDim.x) {
        reinterpret_cast<int4*>(b_shared)[i] = reinterpret_cast<const int4*>(b)[i];
    }
    for (int i = threadIdx.y * 32 + threadIdx.x; i < K / 256; i += blockDim.y * blockDim.x) {
        reinterpret_cast<int4*>(sfb_shared)[i] = reinterpret_cast<const int4*>(sfb)[i];
    }
    __syncthreads();

    // Each warp computes one result and saves it to shared memory
    int result_0 = 0;
    int result_1 = 0;
    int result_2 = 0;
    int result_3 = 0;
    int offset = blockIdx.y * (K * M / 2) + (blockIdx.x * 32 + threadIdx.y) * (K / 2);
    a += offset;
    sfa += offset / 8;
    
    for (int i = threadIdx.x; i < K / 32; i += 32) {
        int4 a_packed = reinterpret_cast<const int4*>(a)[i];
        int4 b_packed = reinterpret_cast<int4*>(b_shared)[i];
        
        __nv_fp8x2_storage_t sfa_fp8x2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(sfa)[i];
        __nv_fp8x2_storage_t sfb_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared)[i];

        multiply_and_accumulate(a_packed, b_packed, sfa_fp8x2, sfb_fp8x2, &result_0, &result_1, &result_2, &result_3);
    }


    // Reduce the result and store it in shared memory
    __half2 reduction_result_0 = __hadd2(reinterpret_cast<const __half2&>(result_0),
            reinterpret_cast<const __half2&>(result_1));
    __half2 reduction_result_1 = __hadd2(reinterpret_cast<const __half2&>(result_2),
            reinterpret_cast<const __half2&>(result_3));
    reduction_result_0 = __hadd2(reduction_result_0, reduction_result_1);
    float final_result_f = __half22float2(reduction_result_0).x + __half22float2(reduction_result_0).y;
    for (int offset = 16; offset > 0; offset /= 2) {
        final_result_f += __shfl_down_sync(FULL_MASK, final_result_f, offset);
    }
    if (threadIdx.x == 0) {
        int c_offset = blockIdx.y * M + blockIdx.x * 32 + threadIdx.y;
        c[c_offset] = __float2half_rn(final_result_f);
    }
}


__global__ void
__launch_bounds__(1024)
gemv_kernel_4096_7168_L8(
    const int4* __restrict__ a,
    const int4* __restrict__ b,
    const int* __restrict__ sfa,
    const int* __restrict__ sfb,
    __half* __restrict__ c
) {
    const int M = 4096;
    const int K = 7168;
    const int K_TILES = 7;      // 7 tiles of 1024 each
    const int TILE_K = 1024;    // elements per tile
    const int Q_SIZE = 2;
    
    // Row distribution: 10 blocks handle 228 rows, 8 blocks handle 227 rows
    // 10 * 228 + 8 * 227 = 2280 + 1816 = 4096
    const int rows_before = (blockIdx.x < 10) ? (blockIdx.x * 228) : (10 * 228 + (blockIdx.x - 10) * 227);
    const int rows_in_block = (blockIdx.x < 10) ? 228 : 227;
    
    // Shared memory for B (all 7 tiles loaded upfront)
    // Each tile: 1024/32 = 32 int4s, 1024/16 = 64 fp8s = 16 ints
    __shared__ int4 b_shared[K_TILES][32];      // [tile][lane]
    __shared__ int sfb_shared[K_TILES][16];     // [tile][16 ints = 64 fp8s]
    
    // Prefetch buffers for A (Q_SIZE + 1 = 3 slots, each warp has its own)
    __shared__ int4 a_shared[Q_SIZE + 1][31][32];   // [buffer][warp][lane]
    __shared__ int sfa_shared[Q_SIZE + 1][31][16];  // [buffer][warp][16 ints]
    
    __shared__ __mbarrier_t bar[K_TILES];
    
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < K_TILES; i++) {
            __mbarrier_init(&bar[i], 32);
        }
    }
    __syncthreads();
    
    // B is laid out as [L, K/2, 128] with 128-padding
    b += blockIdx.y * (K / 2) * 128 / 16;
    sfb += blockIdx.y * (K / 16) * 128 / 4;
    
    // A is laid out as [L, M, K/2]
    const int4* a_base = a + blockIdx.y * (M * K / 32);
    const int* sfa_base = sfa + blockIdx.y * (M * K / 64);
    
    if (threadIdx.y == 0) {
        // ========== WARP 0: Load all B tiles ==========
        #pragma unroll
        for (int tile = 0; tile < K_TILES; tile++) {
            // Each tile: 32 int4s for B, 4 int4s for SFB
            __pipeline_memcpy_async(&b_shared[tile][threadIdx.x], 
                                     &b[tile * 32 + threadIdx.x], sizeof(int4));
            if (tile == K_TILES - 1) {
                if (threadIdx.x < 4) {
                    __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfb_shared[tile])[threadIdx.x], 
                                            &reinterpret_cast<const int4*>(sfb)[tile * 4 + threadIdx.x], sizeof(int4));
                }
            } else if (tile % 2 == 0 && threadIdx.x < 8) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfb_shared[tile])[threadIdx.x], 
                                        &reinterpret_cast<const int4*>(sfb)[tile * 4 + threadIdx.x], sizeof(int4));
            }
            __pipeline_arrive_on(&bar[tile]);
            __mbarrier_arrive(&bar[tile]);
        }
    } else {
        // ========== COMPUTE WARPS (1-31): Process rows in pairs ==========
        const int warp_id = threadIdx.y - 1;  // 0 to 30
        
        // Calculate which rows this warp handles
        const int base_rows_per_warp = rows_in_block / 31;
        const int extra_rows = rows_in_block % 31;
        const int my_rows = base_rows_per_warp + (warp_id < extra_rows ? 1 : 0);
        const int my_first_row = (warp_id < extra_rows) ? 
                                  (warp_id * (base_rows_per_warp + 1)) :
                                  (extra_rows * (base_rows_per_warp + 1) + (warp_id - extra_rows) * base_rows_per_warp);
        
        // Process rows in pairs (following 7168_16384 pattern)
        for (int row_pair = 0; row_pair < my_rows; row_pair += 2) {
            const int local_row0 = row_pair;
            const int local_row1 = row_pair + 1;
            const bool has_row1 = (local_row1 < my_rows);
            
            const int global_row0 = rows_before + my_first_row + local_row0;
            const int global_row1 = has_row1 ? (rows_before + my_first_row + local_row1) : 0;
            
            int result[2][4] = {0};
            
            // Pointers to row data (each row is K/32 = 224 int4s, each tile is 32 int4s)
            const int4* a_row0 = a_base + global_row0 * (K / 32);
            const int* sfa_row0 = sfa_base + global_row0 * (K / 64);
            const int4* a_row1 = has_row1 ? (a_base + global_row1 * (K / 32)) : a_row0;
            const int* sfa_row1 = has_row1 ? (sfa_base + global_row1 * (K / 64)) : sfa_row0;
            
            // Prologue: prefetch tile 0 for both rows
            __pipeline_memcpy_async(&a_shared[0][warp_id][threadIdx.x], 
                                     &a_row0[0 * 32 + threadIdx.x], sizeof(int4));
            if (threadIdx.x < 4) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[0][warp_id])[threadIdx.x], 
                                         &reinterpret_cast<const int4*>(sfa_row0)[0 * 4 + threadIdx.x], sizeof(int4));
            }
            __pipeline_commit();
            
            if (has_row1) {
                __pipeline_memcpy_async(&a_shared[1][warp_id][threadIdx.x], 
                                         &a_row1[0 * 32 + threadIdx.x], sizeof(int4));
                if (threadIdx.x < 4) {
                    __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[1][warp_id])[threadIdx.x], 
                                             &reinterpret_cast<const int4*>(sfa_row1)[0 * 4 + threadIdx.x], sizeof(int4));
                }
            }
            __pipeline_commit();
            
            // Main loop: tiles 0 to K_TILES-2
            #pragma unroll
            for (int tile = 0; tile < K_TILES - 1; tile++) {
                const int next_tile = tile + 1;
                
                // Prefetch row 0 for next tile
                __pipeline_memcpy_async(&a_shared[(tile * 2 + 2) % (Q_SIZE + 1)][warp_id][threadIdx.x], 
                                         &a_row0[next_tile * 32 + threadIdx.x], sizeof(int4));
                if (threadIdx.x < 4) {
                    __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[(tile * 2 + 2) % (Q_SIZE + 1)][warp_id])[threadIdx.x], 
                                             &reinterpret_cast<const int4*>(sfa_row0)[next_tile * 4 + threadIdx.x], sizeof(int4));
                }
                __pipeline_commit();
                
                // Wait for B[tile]
                while (!ptx::mbarrier_try_wait_parity(&bar[tile], 0)) {}
                int4 b_packed = b_shared[tile][threadIdx.x];
                __nv_fp8x2_storage_t sfb_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[tile])[threadIdx.x];
                
                // Wait for A[tile][row0], compute
                __pipeline_wait_prior(Q_SIZE);
                int4 a_packed_r0 = a_shared[(tile * 2) % (Q_SIZE + 1)][warp_id][threadIdx.x];
                __nv_fp8x2_storage_t sfa_fp8x2_r0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(tile * 2) % (Q_SIZE + 1)][warp_id])[threadIdx.x];
                multiply_and_accumulate(a_packed_r0, b_packed, sfa_fp8x2_r0, sfb_fp8x2, 
                                        &result[0][0], &result[0][1], &result[0][2], &result[0][3]);
                
                // Prefetch row 1 for next tile
                if (has_row1) {
                    __pipeline_memcpy_async(&a_shared[(tile * 2 + 3) % (Q_SIZE + 1)][warp_id][threadIdx.x], 
                                             &a_row1[next_tile * 32 + threadIdx.x], sizeof(int4));
                    if (threadIdx.x < 4) {
                        __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[(tile * 2 + 3) % (Q_SIZE + 1)][warp_id])[threadIdx.x], 
                                                 &reinterpret_cast<const int4*>(sfa_row1)[next_tile * 4 + threadIdx.x], sizeof(int4));
                    }
                }
                __pipeline_commit();
                
                // Wait for A[tile][row1], compute
                if (has_row1) {
                    __pipeline_wait_prior(Q_SIZE);
                    int4 a_packed_r1 = a_shared[(tile * 2 + 1) % (Q_SIZE + 1)][warp_id][threadIdx.x];
                    __nv_fp8x2_storage_t sfa_fp8x2_r1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(tile * 2 + 1) % (Q_SIZE + 1)][warp_id])[threadIdx.x];
                    multiply_and_accumulate(a_packed_r1, b_packed, sfa_fp8x2_r1, sfb_fp8x2, 
                                            &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
                }
            }
            
            // Epilogue: last tile (tile = K_TILES - 1)
            {
                const int tile = K_TILES - 1;
                
                while (!ptx::mbarrier_try_wait_parity(&bar[tile], 0)) {}
                int4 b_packed = b_shared[tile][threadIdx.x];
                __nv_fp8x2_storage_t sfb_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[tile])[threadIdx.x];
                
                // Row 0
                __pipeline_wait_prior(1);
                int4 a_packed_r0 = a_shared[(tile * 2) % (Q_SIZE + 1)][warp_id][threadIdx.x];
                __nv_fp8x2_storage_t sfa_fp8x2_r0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(tile * 2) % (Q_SIZE + 1)][warp_id])[threadIdx.x];
                multiply_and_accumulate(a_packed_r0, b_packed, sfa_fp8x2_r0, sfb_fp8x2, 
                                        &result[0][0], &result[0][1], &result[0][2], &result[0][3]);
                
                // Row 1
                if (has_row1) {
                    __pipeline_wait_prior(0);
                    int4 a_packed_r1 = a_shared[(tile * 2 + 1) % (Q_SIZE + 1)][warp_id][threadIdx.x];
                    __nv_fp8x2_storage_t sfa_fp8x2_r1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(tile * 2 + 1) % (Q_SIZE + 1)][warp_id])[threadIdx.x];
                    multiply_and_accumulate(a_packed_r1, b_packed, sfa_fp8x2_r1, sfb_fp8x2, 
                                            &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
                }
            }
            
            // Reduction and store for both rows
            float final_result_f[2];
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                __half2 reduction_result_0 = __hadd2(reinterpret_cast<const __half2&>(result[i][0]),
                        reinterpret_cast<const __half2&>(result[i][1]));
                __half2 reduction_result_1 = __hadd2(reinterpret_cast<const __half2&>(result[i][2]),
                        reinterpret_cast<const __half2&>(result[i][3]));
                reduction_result_0 = __hadd2(reduction_result_0, reduction_result_1);
                final_result_f[i] = __half22float2(reduction_result_0).x + __half22float2(reduction_result_0).y;
                for (int offset = 16; offset > 0; offset /= 2) {
                    final_result_f[i] += __shfl_down_sync(FULL_MASK, final_result_f[i], offset);
                }
            }
            
            if (threadIdx.x == 0) {
                int c_offset = blockIdx.y * M + global_row0;
                c[c_offset] = __float2half_rn(final_result_f[0]);
                if (has_row1) {
                    c[blockIdx.y * M + global_row1] = __float2half_rn(final_result_f[1]);
                }
            }
        }
    }
}


__global__ void
__launch_bounds__(1024)
gemv_kernel_7168_2048_L4(
    const int4* __restrict__ a,
    const int4* __restrict__ b,
    const int* __restrict__ sfa,
    const int* __restrict__ sfb,
    __half* __restrict__ c
) {
    const int M = 7168;
    const int K = 2048;
    
    // Row distribution: 27 blocks handle 194 rows, 10 blocks handle 193 rows
    // 27 * 194 + 10 * 193 = 5238 + 1930 = 7168
    const int rows_before = (blockIdx.x < 27) ? (blockIdx.x * 194) : (27 * 194 + (blockIdx.x - 27) * 193);
    const int rows_in_block = (blockIdx.x < 27) ? 194 : 193;
    
    // Shared memory for B and SFB (loaded once, used by all warps)
    // K/32 = 64 int4s for B, K/16 = 128 fp8s = 32 ints for SFB
    __shared__ int4 b_shared[2][32];      // [half][lane]
    __shared__ int sfb_shared[32];         // 128 fp8s as 32 ints
    
    // Prefetch buffers for A (each warp has its own buffer, double buffered)
    // 31 compute warps (warps 1-31)
    __shared__ int4 a_shared[2][31][2][32];   // [buffer][warp][half][lane]
    __shared__ int sfa_shared[2][31][32];     // [buffer][warp][32 ints = 64 fp8x2]
    
    __shared__ __mbarrier_t bar;
    
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        __mbarrier_init(&bar, 32);
    }
    __syncthreads();
    
    // B is laid out as [L, K/2, 128] with 128-padding
    // Adjust pointers for this L slice (blockIdx.y)
    // Stride in L dimension: (K/2) * 128 bytes = K * 64 bytes
    // As int4 (16 bytes): stride = K * 4
    b += blockIdx.y * (K / 2) * 128 / 16;
    // SFB stride: (K/16) * 128 bytes = K * 8 bytes
    // As int (4 bytes): stride = K * 2
    sfb += blockIdx.y * (K / 16) * 128 / 4;
    
    // A is laid out as [L, M, K/2]
    // Stride in L dimension: M * K/2 bytes
    // As int4: stride = M * K / 32
    const int4* a_base = a + blockIdx.y * (M * K / 32);
    const int* sfa_base = sfa + blockIdx.y * (M * K / 64);
    
    if (threadIdx.y == 0) {
        // ========== WARP 0: Load B and SFB (one-shot, no loop) ==========
        // K/32 = 64 int4s, 32 threads each load 2
        __pipeline_memcpy_async(&b_shared[0][threadIdx.x], &b[threadIdx.x], sizeof(int4));
        __pipeline_memcpy_async(&b_shared[1][threadIdx.x], &b[32 + threadIdx.x], sizeof(int4));
        // K/256 = 8 int4s for SFB
        if (threadIdx.x < 8) {
            __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfb_shared)[threadIdx.x], 
                                     &reinterpret_cast<const int4*>(sfb)[threadIdx.x], sizeof(int4));
        }
        __pipeline_arrive_on(&bar);
        __mbarrier_arrive(&bar);
    } else {
        // ========== COMPUTE WARPS (warps 1-31): Load A/SFA and compute ==========
        const int warp_id = threadIdx.y - 1;  // 0 to 30
        
        // Prologue: prefetch first row for this warp
        int row = warp_id;
        if (row < rows_in_block) {
            int global_row = rows_before + row;
            const int4* a_row = a_base + global_row * (K / 32);  // K/32 int4s per row
            const int* sfa_row = sfa_base + global_row * (K / 64);  // K/64 ints per row
            
            __pipeline_memcpy_async(&a_shared[0][warp_id][0][threadIdx.x], &a_row[threadIdx.x], sizeof(int4));
            __pipeline_memcpy_async(&a_shared[0][warp_id][1][threadIdx.x], &a_row[32 + threadIdx.x], sizeof(int4));
            if (threadIdx.x < 8) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[0][warp_id])[threadIdx.x], 
                                         &reinterpret_cast<const int4*>(sfa_row)[threadIdx.x], sizeof(int4));
            }
        }
        __pipeline_commit();
        
        // Wait for B to be ready
        while (!ptx::mbarrier_try_wait_parity(&bar, 0)) {}
        
        // Load B and SFB from shared to registers (reused for all rows)
        int4 b_packed_0 = b_shared[0][threadIdx.x];
        int4 b_packed_1 = b_shared[1][threadIdx.x];
        __nv_fp8x2_storage_t sfb_fp8x2_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared)[threadIdx.x];
        __nv_fp8x2_storage_t sfb_fp8x2_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared)[threadIdx.x + 32];
        
        // Main loop: iterate over assigned rows with stride 31
        for (int row = warp_id; row < rows_in_block; row += 31) {
            int next_row = row + 31;
            int buf = (row / 31) & 1;
            int next_buf = 1 - buf;
            
            // Prefetch next row if exists
            if (next_row < rows_in_block) {
                int global_next_row = rows_before + next_row;
                const int4* a_row = a_base + global_next_row * (K / 32);
                const int* sfa_row = sfa_base + global_next_row * (K / 64);
                
                __pipeline_memcpy_async(&a_shared[next_buf][warp_id][0][threadIdx.x], &a_row[threadIdx.x], sizeof(int4));
                __pipeline_memcpy_async(&a_shared[next_buf][warp_id][1][threadIdx.x], &a_row[32 + threadIdx.x], sizeof(int4));
                if (threadIdx.x < 8) {
                    __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[next_buf][warp_id])[threadIdx.x], 
                                             &reinterpret_cast<const int4*>(sfa_row)[threadIdx.x], sizeof(int4));
                }
            }
            __pipeline_commit();
            
            // Wait for current row's A data
            __pipeline_wait_prior(1);
            
            // Load A from shared
            int4 a_packed_0 = a_shared[buf][warp_id][0][threadIdx.x];
            int4 a_packed_1 = a_shared[buf][warp_id][1][threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[buf][warp_id])[threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[buf][warp_id])[threadIdx.x + 32];
            
            // Compute: multiply_and_accumulate for both halves
            int result[4] = {0, 0, 0, 0};
            multiply_and_accumulate(a_packed_0, b_packed_0, sfa_fp8x2_0, sfb_fp8x2_0, 
                                    &result[0], &result[1], &result[2], &result[3]);
            multiply_and_accumulate(a_packed_1, b_packed_1, sfa_fp8x2_1, sfb_fp8x2_1, 
                                    &result[0], &result[1], &result[2], &result[3]);
            
            // Reduce and store
            __half2 reduction_result_0 = __hadd2(reinterpret_cast<const __half2&>(result[0]),
                    reinterpret_cast<const __half2&>(result[1]));
            __half2 reduction_result_1 = __hadd2(reinterpret_cast<const __half2&>(result[2]),
                    reinterpret_cast<const __half2&>(result[3]));
            reduction_result_0 = __hadd2(reduction_result_0, reduction_result_1);
            float final_result_f = __half22float2(reduction_result_0).x + __half22float2(reduction_result_0).y;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                final_result_f += __shfl_down_sync(FULL_MASK, final_result_f, offset);
            }
            
            if (threadIdx.x == 0) {
                int global_row = rows_before + row;
                int c_offset = blockIdx.y * M + global_row;
                c[c_offset] = __float2half_rn(final_result_f);
            }
        }
    }
}



__global__ void
__launch_bounds__(832)
gemv_kernel_7168_16384(
    const int4* __restrict__ a,
    const int4* __restrict__ b,
    const int* __restrict__ sfa,
    const int* __restrict__ sfb,
    __half* __restrict__ c
) {
    const int M = 7168;
    const int K = 16384;
    const int Q_SIZE = 2;
    const int active_warps = (blockIdx.x < 32) ? 26 : 25;

    __shared__ int4 a_shared[Q_SIZE + 1][25][2][32];
    __shared__ int sfa_shared[Q_SIZE + 1][25][32];

    // We will load all b and sfb, because we can, it simplifies the logic
    __shared__ int4 b_shared[8][2][32];
    __shared__ int sfb_shared[8][32];

    __shared__ __mbarrier_t bar[8];
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            __mbarrier_init(&bar[i], 32);
        }
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        // ========== WARP 0: Load b and sfb for all columns ==========
        #pragma unroll
        for (int col_idx = 0; col_idx < 8; col_idx++) {
            __pipeline_memcpy_async(&b_shared[col_idx][0][threadIdx.x], &b[col_idx * 64 + threadIdx.x], sizeof(int4));
            __pipeline_memcpy_async(&b_shared[col_idx][1][threadIdx.x], &b[col_idx * 64 + 32 + threadIdx.x], sizeof(int4));
            if (threadIdx.x < 8) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfb_shared[col_idx])[threadIdx.x], &reinterpret_cast<const int4*>(sfb)[col_idx * 8 + threadIdx.x], sizeof(int4));
            }
            __pipeline_arrive_on(&bar[col_idx]);
            __mbarrier_arrive(&bar[col_idx]);
        }
    } else if (threadIdx.y < active_warps) {
        // ========== COMPUTE WARPS: Load a/sfa and compute ==========
        int offset = (blockIdx.x * 24 + min(blockIdx.x, 32) + threadIdx.y - 1) * 2 * (K / 2);
        a += offset / 16;
        sfa += offset / 32;

        int result[2][4] = {0};

        // Prologue: prefetch col 0 (both rows)
        __pipeline_memcpy_async(&a_shared[0][threadIdx.y - 1][0][threadIdx.x], &a[0 * (K / 32) + 0 * 64 + threadIdx.x], sizeof(int4));
        __pipeline_memcpy_async(&a_shared[0][threadIdx.y - 1][1][threadIdx.x], &a[0 * (K / 32) + 0 * 64 + 32 + threadIdx.x], sizeof(int4));
        if (threadIdx.x < 8) {
            __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[0][threadIdx.y - 1])[threadIdx.x], &reinterpret_cast<const int4*>(sfa)[0 * (K / 256) + 0 * 8 + threadIdx.x], sizeof(int4));
        }
        __pipeline_commit();
        __pipeline_memcpy_async(&a_shared[1][threadIdx.y - 1][0][threadIdx.x], &a[1 * (K / 32) + 0 * 64 + threadIdx.x], sizeof(int4));
        __pipeline_memcpy_async(&a_shared[1][threadIdx.y - 1][1][threadIdx.x], &a[1 * (K / 32) + 0 * 64 + 32 + threadIdx.x], sizeof(int4));
        if (threadIdx.x < 8) {
            __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[1][threadIdx.y - 1])[threadIdx.x], &reinterpret_cast<const int4*>(sfa)[1 * (K / 256) + 0 * 8 + threadIdx.x], sizeof(int4));
        }
        __pipeline_commit();

        // Main loop: process columns 0-6, prefetch next column
        #pragma unroll
        for (int col_idx = 0; col_idx < 7; col_idx++) {
            int next_col = col_idx + 1;

            // Prefetch row 0 for next column
            __pipeline_memcpy_async(&a_shared[(col_idx * 2 + 2) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x], &a[0 * (K / 32) + next_col * 64 + threadIdx.x], sizeof(int4));
            __pipeline_memcpy_async(&a_shared[(col_idx * 2 + 2) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x], &a[0 * (K / 32) + next_col * 64 + 32 + threadIdx.x], sizeof(int4));
            if (threadIdx.x < 8) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[(col_idx * 2 + 2) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x], &reinterpret_cast<const int4*>(sfa)[0 * (K / 256) + next_col * 8 + threadIdx.x], sizeof(int4));
            }
            __pipeline_commit();

            // Wait for b/sfb data, load once for this column
            while (!ptx::mbarrier_try_wait_parity(&bar[col_idx], 0)) {}
            int4 b_packed_0 = b_shared[col_idx][0][threadIdx.x];
            int4 b_packed_1 = b_shared[col_idx][1][threadIdx.x];
            __nv_fp8x2_storage_t sfb_fp8x2_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[col_idx])[threadIdx.x];
            __nv_fp8x2_storage_t sfb_fp8x2_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[col_idx])[threadIdx.x + 32];

            // Load and compute row 0
            __pipeline_wait_prior(Q_SIZE);
            int4 a_packed_r0_0 = a_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x];
            int4 a_packed_r0_1 = a_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r0_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r0_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x + 32];
            multiply_and_accumulate(a_packed_r0_0, b_packed_0, sfa_fp8x2_r0_0, sfb_fp8x2_0, &result[0][0], &result[0][1], &result[0][2], &result[0][3]);
            multiply_and_accumulate(a_packed_r0_1, b_packed_1, sfa_fp8x2_r0_1, sfb_fp8x2_1, &result[0][0], &result[0][1], &result[0][2], &result[0][3]);

            // Prefetch row 1 for next column
            __pipeline_memcpy_async(&a_shared[(col_idx * 2 + 3) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x], &a[1 * (K / 32) + next_col * 64 + threadIdx.x], sizeof(int4));
            __pipeline_memcpy_async(&a_shared[(col_idx * 2 + 3) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x], &a[1 * (K / 32) + next_col * 64 + 32 + threadIdx.x], sizeof(int4));
            if (threadIdx.x < 8) {
                __pipeline_memcpy_async(&reinterpret_cast<int4*>(sfa_shared[(col_idx * 2 + 3) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x], &reinterpret_cast<const int4*>(sfa)[1 * (K / 256) + next_col * 8 + threadIdx.x], sizeof(int4));
            }
            __pipeline_commit();

            // Load and compute row 1
            __pipeline_wait_prior(Q_SIZE);
            int4 a_packed_r1_0 = a_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x];
            int4 a_packed_r1_1 = a_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r1_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r1_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x + 32];
            multiply_and_accumulate(a_packed_r1_0, b_packed_0, sfa_fp8x2_r1_0, sfb_fp8x2_0, &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
            multiply_and_accumulate(a_packed_r1_1, b_packed_1, sfa_fp8x2_r1_1, sfb_fp8x2_1, &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
        }

        // Epilogue: process last column (col_idx = 7)
        {
            const int col_idx = 7;

            // Wait for b/sfb data, load once
            while (!ptx::mbarrier_try_wait_parity(&bar[col_idx], 0)) {}
            int4 b_packed_0 = b_shared[col_idx][0][threadIdx.x];
            int4 b_packed_1 = b_shared[col_idx][1][threadIdx.x];
            __nv_fp8x2_storage_t sfb_fp8x2_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[col_idx])[threadIdx.x];
            __nv_fp8x2_storage_t sfb_fp8x2_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared[col_idx])[threadIdx.x + 32];

            // Load and compute row 0
            __pipeline_wait_prior(1);
            int4 a_packed_r0_0 = a_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x];
            int4 a_packed_r0_1 = a_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r0_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r0_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x + 32];
            multiply_and_accumulate(a_packed_r0_0, b_packed_0, sfa_fp8x2_r0_0, sfb_fp8x2_0, &result[0][0], &result[0][1], &result[0][2], &result[0][3]);
            multiply_and_accumulate(a_packed_r0_1, b_packed_1, sfa_fp8x2_r0_1, sfb_fp8x2_1, &result[0][0], &result[0][1], &result[0][2], &result[0][3]);

            // Load and compute row 1
            __pipeline_wait_prior(0);
            int4 a_packed_r1_0 = a_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1][0][threadIdx.x];
            int4 a_packed_r1_1 = a_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1][1][threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r1_0 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x];
            __nv_fp8x2_storage_t sfa_fp8x2_r1_1 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfa_shared[(col_idx * 2 + 1) % (Q_SIZE + 1)][threadIdx.y - 1])[threadIdx.x + 32];
            multiply_and_accumulate(a_packed_r1_0, b_packed_0, sfa_fp8x2_r1_0, sfb_fp8x2_0, &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
            multiply_and_accumulate(a_packed_r1_1, b_packed_1, sfa_fp8x2_r1_1, sfb_fp8x2_1, &result[1][0], &result[1][1], &result[1][2], &result[1][3]);
        }

        // Reduction and store
        float final_result_f[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            __half2 reduction_result_0 = __hadd2(reinterpret_cast<const __half2&>(result[i][0]),
                    reinterpret_cast<const __half2&>(result[i][1]));
            __half2 reduction_result_1 = __hadd2(reinterpret_cast<const __half2&>(result[i][2]),
                    reinterpret_cast<const __half2&>(result[i][3]));
            reduction_result_0 = __hadd2(reduction_result_0, reduction_result_1);
            final_result_f[i] = __half22float2(reduction_result_0).x + __half22float2(reduction_result_0).y;
            for (int offset = 16; offset > 0; offset /= 2) {
                final_result_f[i] += __shfl_down_sync(FULL_MASK, final_result_f[i], offset);
            }
        }
        if (threadIdx.x == 0) {
            __half final_result[2];
            final_result[0] = __float2half_rn(final_result_f[0]);
            final_result[1] = __float2half_rn(final_result_f[1]);
            int c_offset = (blockIdx.x * 24 + min((int)blockIdx.x, 32) + threadIdx.y - 1);
            reinterpret_cast<int*>(c)[c_offset] = reinterpret_cast<int&>(final_result);
        }
    }
}



__global__ void gemv_kernel(
    const __nv_fp4x2_storage_t* __restrict__ a,
    const __nv_fp4x2_storage_t* __restrict__ b,
    const __nv_fp8_e4m3* __restrict__ sfa,
    const __nv_fp8_e4m3* __restrict__ sfb,
    __half* __restrict__ c,
    int M,
    int K
) {
    extern __shared__ unsigned char shared_storage[];
    auto* b_shared = reinterpret_cast<__nv_fp4x2_storage_t*>(shared_storage);
    auto* sfb_shared = reinterpret_cast<__nv_fp8_e4m3*>(b_shared + (K / 2));
    __shared__ __half c_shared[32];

    b += blockIdx.y * (K / 2) * 128;
    sfb += blockIdx.y * (K / 16) * 128;

    for (int i = threadIdx.y * 32 + threadIdx.x; i < K / 32; i += blockDim.y * blockDim.x) {
        reinterpret_cast<int4*>(b_shared)[i] = reinterpret_cast<const int4*>(b)[i];
    }
    for (int i = threadIdx.y * 32 + threadIdx.x; i < K / 256; i += blockDim.y * blockDim.x) {
        reinterpret_cast<int4*>(sfb_shared)[i] = reinterpret_cast<const int4*>(sfb)[i];
    }
    __syncthreads();

    // Each warp computes one result and saves it to shared memory
    int result_0 = 0;
    int result_1 = 0;
    int result_2 = 0;
    int result_3 = 0;
    int offset = blockIdx.y * (K * M / 2) + (blockIdx.x * 32 + threadIdx.y) * (K / 2);
    a += offset;
    sfa += offset / 8;
    
    for (int i = threadIdx.x; i < K / 32; i += 32) {
        int4 a_packed = reinterpret_cast<const int4*>(a)[i];
        int4 b_packed = reinterpret_cast<int4*>(b_shared)[i];
        
        __nv_fp8x2_storage_t sfa_fp8x2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(sfa)[i];
        __nv_fp8x2_storage_t sfb_fp8x2 = reinterpret_cast<__nv_fp8x2_storage_t*>(sfb_shared)[i];

        multiply_and_accumulate(a_packed, b_packed, sfa_fp8x2, sfb_fp8x2, &result_0, &result_1, &result_2, &result_3);
    }


    // Reduce the result and store it in shared memory
    __half2 reduction_result_0 = __hadd2(reinterpret_cast<const __half2&>(result_0),
            reinterpret_cast<const __half2&>(result_1));
    __half2 reduction_result_1 = __hadd2(reinterpret_cast<const __half2&>(result_2),
            reinterpret_cast<const __half2&>(result_3));
    reduction_result_0 = __hadd2(reduction_result_0, reduction_result_1);
    float final_result_f = __half22float2(reduction_result_0).x + __half22float2(reduction_result_0).y;
    for (int offset = 16; offset > 0; offset /= 2) {
        final_result_f += __shfl_down_sync(FULL_MASK, final_result_f, offset);
    }
    if (threadIdx.x == 0) {
        c_shared[threadIdx.y] = __float2half_rn(final_result_f);
    }
    __syncthreads();
    
    // Write the result to global memory
    if (threadIdx.y == 0) {
        int c_offset = blockIdx.y * M + blockIdx.x * 32 + threadIdx.x;
        c[c_offset] = c_shared[threadIdx.x];
    }
}



torch::Tensor gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {
    const int64_t M = a.size(0);
    const int64_t K = a.size(1) * 2;
    const int64_t L = a.size(2);


    dim3 block_dim(32, 32, 1);
    dim3 grid_dim(M / 32, L, 1);
    const auto* a_ptr = reinterpret_cast<const __nv_fp4x2_storage_t*>(a.data_ptr());
    const auto* b_ptr = reinterpret_cast<const __nv_fp4x2_storage_t*>(b.data_ptr());
    const auto* sfa_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(sfa.data_ptr());
    const auto* sfb_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(sfb.data_ptr());
    auto* c_ptr = reinterpret_cast<__half*>(c.data_ptr<c10::Half>());

    size_t shared_mem_bytes =
        (static_cast<size_t>(K) / 2) * sizeof(__nv_fp4x2_storage_t) +
        (static_cast<size_t>(K) / 16) * sizeof(__nv_fp8_e4m3);
    
    if (M == 4096 && K == 7168 && L == 8) {
        grid_dim = dim3(18, 8, 1);
        block_dim = dim3(32, 32, 1);
        gemv_kernel_4096_7168_L8<<<grid_dim, block_dim>>>(
            reinterpret_cast<const int4*>(a.data_ptr()),
            reinterpret_cast<const int4*>(b.data_ptr()),
            reinterpret_cast<const int*>(sfa.data_ptr()),
            reinterpret_cast<const int*>(sfb.data_ptr()),
            c_ptr
        );
    } else if (M == 4096 && K == 7168) {
        gemv_kernel_4096_7168<<<grid_dim, block_dim, shared_mem_bytes>>>(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr
        );
    } else if (M == 7168 && K == 2048 && L == 4) {
        grid_dim = dim3(37, 4, 1);
        block_dim = dim3(32, 32, 1);
        gemv_kernel_7168_2048_L4<<<grid_dim, block_dim>>>(
            reinterpret_cast<const int4*>(a.data_ptr()),
            reinterpret_cast<const int4*>(b.data_ptr()),
            reinterpret_cast<const int*>(sfa.data_ptr()),
            reinterpret_cast<const int*>(sfb.data_ptr()),
            c_ptr
        );
    } else if (M == 7168 && K == 16384) {
        grid_dim = dim3(148, 1, 1);
        block_dim = dim3(32, 26, 1);
        gemv_kernel_7168_16384<<<grid_dim, block_dim>>>(
            reinterpret_cast<const int4*>(a.data_ptr()),
            reinterpret_cast<const int4*>(b.data_ptr()),
            reinterpret_cast<const int*>(sfa.data_ptr()),
            reinterpret_cast<const int*>(sfb.data_ptr()),
            c_ptr
        );
    } else {
        gemv_kernel<<<grid_dim, block_dim, shared_mem_bytes>>>(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            c_ptr,
            static_cast<int>(M),
            static_cast<int>(K)
        );
    }
    return c;
}
"""


cpp_source = """
#include <torch/extension.h>

torch::Tensor gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);
"""

gemv_module = load_inline(
    name='gemv_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gemv_cuda'],
    verbose=True,
    extra_cuda_cflags=['-arch=compute_100a', '-code=sm_100a', '-O3'],
)




def custom_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMV.
    """

    a, b, sfa, sfb, _, _, c = data

    return gemv_module.gemv_cuda(a, b, sfa, sfb, c)
