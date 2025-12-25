# Optimizations:
# 1. Use PTX code aggressively:
# PTX: In process_tile_fused:
# each tile has 64 fp4 (TILE_K), each call of process_tile_fused input 8 int32,
# which contains the 64 fp4 data, each int32 contains 8 fp4 data, each int32 sfa_packed
# contains 4 scale factor in fp8. Because every 16 fp4 share same fp8 scale factor, so we
# 4 groups of 16 fp4 calculation, so the PTX code has 4 groups of similar code.
# 2. Instruction level parallelism: process 2 tiles per thread in one loop. 
# Reordered instructions in loop: precompute offsets/pointers upfront, issue all loads
# early to overlap with computation
# 3. Staggered memory loading: Issue loads in interleaved order (scale factors, then data)
#    to avoid saturating the memory pipeline. The original tile pattern (tidx and tidx+16)
#    is preserved as it provides better instruction-level parallelism - the separated tiles
#    give memory loads more time to complete before being used.
# 4. vectorized 
# 5. coalesced memory access.
# 6. Tuned parameters and register usage.
# 7. warp level reduction for get sum.
# 8. unroll for loop adjust 

# Other tries but not make it faster:
# - use smem for B
# - use smem for C write so we can write in coalesced way
# - double buffer and async copy
# - union float4 to uint32_t for efficient access.

# Next tries idea:
# - 
import torch
import sys
from torch.utils.cpp_extension import load_inline
from typing import Tuple

gemv_cuda_src = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ void process_tile_fused(
    float& local_sum,
    const uint32_t A0_0, const uint32_t A0_1, const uint32_t A0_2, const uint32_t A0_3,
    const uint32_t A1_0, const uint32_t A1_1, const uint32_t A1_2, const uint32_t A1_3,
    const uint32_t B0_0, const uint32_t B0_1, const uint32_t B0_2, const uint32_t B0_3,
    const uint32_t B1_0, const uint32_t B1_1, const uint32_t B1_2, const uint32_t B1_3,
    const uint32_t sfa_packed,
    const uint32_t sfb_packed)
{
    asm volatile (
        "{"
        "  .reg .b16 %%sfalo, %%sfahi, %%sfblo, %%sfbhi;\\n"
        "  .reg .b32 %%sa01, %%sa23, %%sb01, %%sb23;\\n"
        "  .reg .b32 %%scale01, %%scale23;\\n"
        "  .reg .f32 %%s0, %%s1, %%s2, %%s3;\\n"
        "  .reg .b8 %%a<4>, %%b<4>;\\n"
        "  .reg .b32 %%fa<4>, %%fb<4>;\\n"
        "  .reg .b32 %%p0, %%p1, %%p2, %%p3;\\n"
        "  .reg .f16 %%h0, %%h1;\\n"
        "  .reg .f32 %%f0, %%f1, %%acc0, %%acc1, %%acc2, %%acc3, %%tile_result, %%one;\\n"
        
        "  mov.f32 %%one, 0f3f800000;\\n"
        "  mov.b32 {%%sfalo, %%sfahi}, %17;\\n"
        "  mov.b32 {%%sfblo, %%sfbhi}, %18;\\n"
        "  cvt.rn.f16x2.e4m3x2 %%sa01, %%sfalo;\\n"
        "  cvt.rn.f16x2.e4m3x2 %%sa23, %%sfahi;\\n"
        "  cvt.rn.f16x2.e4m3x2 %%sb01, %%sfblo;\\n"
        "  cvt.rn.f16x2.e4m3x2 %%sb23, %%sfbhi;\\n"
        "  mul.rn.f16x2 %%scale01, %%sa01, %%sb01;\\n"
        "  mul.rn.f16x2 %%scale23, %%sa23, %%sb23;\\n"

        "  mov.b32 {%%h0, %%h1}, %%scale01;\\n"
        "  cvt.f32.f16 %%s0, %%h0;\\n"
        "  cvt.f32.f16 %%s1, %%h1;\\n"
        "  mov.b32 {%%h0, %%h1}, %%scale23;\\n"
        "  cvt.f32.f16 %%s2, %%h0;\\n"
        "  cvt.f32.f16 %%s3, %%h1;\\n"
        
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %1;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %9;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  mul.rn.f16x2 %%p0, %%fa0, %%fb0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa1, %%fb1, %%p0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa2, %%fb2, %%p0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa3, %%fb3, %%p0;\\n"

        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %2;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %10;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  fma.rn.f16x2 %%p0, %%fa0, %%fb0, %%p0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa1, %%fb1, %%p0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa2, %%fb2, %%p0;\\n"
        "  fma.rn.f16x2 %%p0, %%fa3, %%fb3, %%p0;\\n"
        "  mov.b32 {%%h0, %%h1}, %%p0;\\n"
        "  cvt.f32.f16 %%f0, %%h0;\\n"
        "  cvt.f32.f16 %%f1, %%h1;\\n"
        "  add.f32 %%acc0, %%f0, %%f1;\\n"
        "  mul.f32 %%acc0, %%acc0, %%s0;\\n"
        
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %3;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %11;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  mul.rn.f16x2 %%p1, %%fa0, %%fb0;\\n"
        "  fma.rn.f16x2 %%p1, %%fa1, %%fb1, %%p1;\\n"
        "  fma.rn.f16x2 %%p1, %%fa2, %%fb2, %%p1;\\n"
        "  fma.rn.f16x2 %%p1, %%fa3, %%fb3, %%p1;\\n"
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %4;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %12;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  fma.rn.f16x2 %%p1, %%fa0, %%fb0, %%p1;\\n"
        "  fma.rn.f16x2 %%p1, %%fa1, %%fb1, %%p1;\\n"
        "  fma.rn.f16x2 %%p1, %%fa2, %%fb2, %%p1;\\n"
        "  fma.rn.f16x2 %%p1, %%fa3, %%fb3, %%p1;\\n"
        "  mov.b32 {%%h0, %%h1}, %%p1;\\n"
        "  cvt.f32.f16 %%f0, %%h0;\\n"
        "  cvt.f32.f16 %%f1, %%h1;\\n"
        "  add.f32 %%acc1, %%f0, %%f1;\\n"
        "  fma.rn.f32 %%acc0, %%acc1, %%s1, %%acc0;\\n"
        
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %5;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %13;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  mul.rn.f16x2 %%p2, %%fa0, %%fb0;\\n"
        "  fma.rn.f16x2 %%p2, %%fa1, %%fb1, %%p2;\\n"
        "  fma.rn.f16x2 %%p2, %%fa2, %%fb2, %%p2;\\n"
        "  fma.rn.f16x2 %%p2, %%fa3, %%fb3, %%p2;\\n"
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %6;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %14;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  fma.rn.f16x2 %%p2, %%fa0, %%fb0, %%p2;\\n"
        "  fma.rn.f16x2 %%p2, %%fa1, %%fb1, %%p2;\\n"
        "  fma.rn.f16x2 %%p2, %%fa2, %%fb2, %%p2;\\n"
        "  fma.rn.f16x2 %%p2, %%fa3, %%fb3, %%p2;\\n"
        "  mov.b32 {%%h0, %%h1}, %%p2;\\n"
        "  cvt.f32.f16 %%f0, %%h0;\\n"
        "  cvt.f32.f16 %%f1, %%h1;\\n"
        "  add.f32 %%acc2, %%f0, %%f1;\\n"
        "  fma.rn.f32 %%acc0, %%acc2, %%s2, %%acc0;\\n"
        
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %7;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %15;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  mul.rn.f16x2 %%p3, %%fa0, %%fb0;\\n"
        "  fma.rn.f16x2 %%p3, %%fa1, %%fb1, %%p3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa2, %%fb2, %%p3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa3, %%fb3, %%p3;\\n"
        "  mov.b32 {%%a0, %%a1, %%a2, %%a3}, %8;\\n"
        "  mov.b32 {%%b0, %%b1, %%b2, %%b3}, %16;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa0, %%a0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa1, %%a1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa2, %%a2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fa3, %%a3;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb0, %%b0;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb1, %%b1;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb2, %%b2;\\n"
        "  cvt.rn.f16x2.e2m1x2 %%fb3, %%b3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa0, %%fb0, %%p3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa1, %%fb1, %%p3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa2, %%fb2, %%p3;\\n"
        "  fma.rn.f16x2 %%p3, %%fa3, %%fb3, %%p3;\\n"
        "  mov.b32 {%%h0, %%h1}, %%p3;\\n"
        "  cvt.f32.f16 %%f0, %%h0;\\n"
        "  cvt.f32.f16 %%f1, %%h1;\\n"
        "  add.f32 %%acc3, %%f0, %%f1;\\n"
        "  fma.rn.f32 %%tile_result, %%acc3, %%s3, %%acc0;\\n"
        "  fma.rn.f32 %0, %%tile_result, %%one, %0;\\n"
        "}"
        : "+f"(local_sum)
        : "r"(A0_0), "r"(A0_1), "r"(A0_2), "r"(A0_3),
          "r"(A1_0), "r"(A1_1), "r"(A1_2), "r"(A1_3),
          "r"(B0_0), "r"(B0_1), "r"(B0_2), "r"(B0_3),
          "r"(B1_0), "r"(B1_1), "r"(B1_2), "r"(B1_3),
          "r"(sfa_packed), "r"(sfb_packed)
    );
}

extern "C" __global__ __launch_bounds__(128, 8)
void block_scaled_gemv_fp4_fp8_fp16(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ SFA,
    const uint8_t* __restrict__ SFB,
    __half* __restrict__ C,
    int M, int K, int L)
{
    constexpr int ROWS_PER_BLOCK = 8;
    constexpr int THREADS_PER_ROW = 16;
    constexpr int TILE_K = 64;

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int block_row = blockIdx.x * ROWS_PER_BLOCK;
    const int batch_idx = blockIdx.z;
    const int global_row = block_row + tidy;

    // Predicate for valid row - use throughout instead of early return
    const bool valid_row = global_row < M;

    const int num_k_tiles = K >> 6;
    
    // Calculate iterations such that ALL threads do the same number
    // Process in pairs where possible
    const int num_paired_iters = (num_k_tiles / (2 * THREADS_PER_ROW));
    const int remaining_after_pairs = num_k_tiles - (num_paired_iters * 2 * THREADS_PER_ROW);
    
    // Check if there's a full iteration where all 16 threads work
    const bool has_single_iter = (remaining_after_pairs >= THREADS_PER_ROW);

    const uint8_t* B_base = B + batch_idx * (128 * (K >> 1));
    const uint8_t* SFB_base = SFB + batch_idx * (128 * (K >> 4));
    const uint8_t* A_row = A + batch_idx * (M * (K >> 1)) + global_row * (K >> 1);
    const uint8_t* SFA_row = SFA + batch_idx * (M * (K >> 4)) + global_row * (K >> 4);

    float local_sum = 0.0f;

    // All threads execute same number of paired iterations
    #pragma unroll 1
    for (int iter = 0; iter < num_paired_iters; ++iter) {
        const int tile = tidx + iter * 2 * THREADS_PER_ROW;
        
        const int k_offset_0_half = (tile * TILE_K) >> 1;
        const int k_offset_1_half = ((tile + THREADS_PER_ROW) * TILE_K) >> 1;
        const int k_offset_0_quarter = (tile * TILE_K) >> 4;
        const int k_offset_1_quarter = ((tile + THREADS_PER_ROW) * TILE_K) >> 4;
        
        const float4* A_ptr_0 = reinterpret_cast<const float4*>(A_row + k_offset_0_half);
        const float4* B_ptr_0 = reinterpret_cast<const float4*>(B_base + k_offset_0_half);
        const float4* A_ptr_1 = reinterpret_cast<const float4*>(A_row + k_offset_1_half);
        const float4* B_ptr_1 = reinterpret_cast<const float4*>(B_base + k_offset_1_half);
        const uint32_t* sfa_ptr_0 = reinterpret_cast<const uint32_t*>(SFA_row + k_offset_0_quarter);
        const uint32_t* sfb_ptr_0 = reinterpret_cast<const uint32_t*>(SFB_base + k_offset_0_quarter);
        const uint32_t* sfa_ptr_1 = reinterpret_cast<const uint32_t*>(SFA_row + k_offset_1_quarter);
        const uint32_t* sfb_ptr_1 = reinterpret_cast<const uint32_t*>(SFB_base + k_offset_1_quarter);
        
        // Load all data first
        uint32_t sfa_t0 = __ldg(sfa_ptr_0);
        uint32_t sfb_t0 = __ldg(sfb_ptr_0);
        uint32_t sfa_t1 = __ldg(sfa_ptr_1);
        uint32_t sfb_t1 = __ldg(sfb_ptr_1);
        
        float4 A_data0_t0 = __ldg(A_ptr_0);
        float4 B_data0_t0 = __ldg(B_ptr_0);
        float4 A_data1_t0 = __ldg(A_ptr_0 + 1);
        float4 B_data1_t0 = __ldg(B_ptr_0 + 1);
        
        float4 A_data0_t1 = __ldg(A_ptr_1);
        float4 B_data0_t1 = __ldg(B_ptr_1);
        float4 A_data1_t1 = __ldg(A_ptr_1 + 1);
        float4 B_data1_t1 = __ldg(B_ptr_1 + 1);

        const uint32_t* A0_u32_t0 = reinterpret_cast<const uint32_t*>(&A_data0_t0);
        const uint32_t* A1_u32_t0 = reinterpret_cast<const uint32_t*>(&A_data1_t0);
        const uint32_t* B0_u32_t0 = reinterpret_cast<const uint32_t*>(&B_data0_t0);
        const uint32_t* B1_u32_t0 = reinterpret_cast<const uint32_t*>(&B_data1_t0);
        const uint32_t* A0_u32_t1 = reinterpret_cast<const uint32_t*>(&A_data0_t1);
        const uint32_t* A1_u32_t1 = reinterpret_cast<const uint32_t*>(&A_data1_t1);
        const uint32_t* B0_u32_t1 = reinterpret_cast<const uint32_t*>(&B_data0_t1);
        const uint32_t* B1_u32_t1 = reinterpret_cast<const uint32_t*>(&B_data1_t1);
        
        process_tile_fused(
            local_sum,
            A0_u32_t0[0], A0_u32_t0[1], A0_u32_t0[2], A0_u32_t0[3],
            A1_u32_t0[0], A1_u32_t0[1], A1_u32_t0[2], A1_u32_t0[3],
            B0_u32_t0[0], B0_u32_t0[1], B0_u32_t0[2], B0_u32_t0[3],
            B1_u32_t0[0], B1_u32_t0[1], B1_u32_t0[2], B1_u32_t0[3],
            sfa_t0, sfb_t0);

        process_tile_fused(
            local_sum,
            A0_u32_t1[0], A0_u32_t1[1], A0_u32_t1[2], A0_u32_t1[3],
            A1_u32_t1[0], A1_u32_t1[1], A1_u32_t1[2], A1_u32_t1[3],
            B0_u32_t1[0], B0_u32_t1[1], B0_u32_t1[2], B0_u32_t1[3],
            B1_u32_t1[0], B1_u32_t1[1], B1_u32_t1[2], B1_u32_t1[3],
            sfa_t1, sfb_t1);
    }
    
    // Single iteration block - all threads execute if has work
    // All 16 threads participate - no divergence
    if (has_single_iter) {
        const int tile = tidx + num_paired_iters * 2 * THREADS_PER_ROW;
        const int k_offset = tile * TILE_K;
        const float4* A_ptr = reinterpret_cast<const float4*>(A_row + (k_offset >> 1));
        const float4* B_ptr = reinterpret_cast<const float4*>(B_base + (k_offset >> 1));

        float4 A_data0 = __ldg(A_ptr);
        float4 B_data0 = __ldg(B_ptr);
        float4 A_data1 = __ldg(A_ptr + 1);
        float4 B_data1 = __ldg(B_ptr + 1);
        uint32_t sfa_vec = __ldg(reinterpret_cast<const uint32_t*>(SFA_row + (k_offset >> 4)));
        uint32_t sfb_vec = __ldg(reinterpret_cast<const uint32_t*>(SFB_base + (k_offset >> 4)));

        const uint32_t* A0_u32 = reinterpret_cast<const uint32_t*>(&A_data0);
        const uint32_t* A1_u32 = reinterpret_cast<const uint32_t*>(&A_data1);
        const uint32_t* B0_u32 = reinterpret_cast<const uint32_t*>(&B_data0);
        const uint32_t* B1_u32 = reinterpret_cast<const uint32_t*>(&B_data1);

        process_tile_fused(
            local_sum,
            A0_u32[0], A0_u32[1], A0_u32[2], A0_u32[3],
            A1_u32[0], A1_u32[1], A1_u32[2], A1_u32[3],
            B0_u32[0], B0_u32[1], B0_u32[2], B0_u32[3],
            B1_u32[0], B1_u32[1], B1_u32[2], B1_u32[3],
            sfa_vec, sfb_vec);
    }
    
    // Final stragglers (< 16 tiles remaining)
    // Accept minimal divergence here - it's unavoidable for remainder
    // This only happens when num_k_tiles % THREADS_PER_ROW != 0
    const int final_start = num_paired_iters * 2 * THREADS_PER_ROW + 
                            (has_single_iter ? THREADS_PER_ROW : 0);
    if (tidx + final_start < num_k_tiles) {
        const int tile = tidx + final_start;
        const int k_offset = tile * TILE_K;
        const float4* A_ptr = reinterpret_cast<const float4*>(A_row + (k_offset >> 1));
        const float4* B_ptr = reinterpret_cast<const float4*>(B_base + (k_offset >> 1));

        float4 A_data0 = __ldg(A_ptr);
        float4 B_data0 = __ldg(B_ptr);
        float4 A_data1 = __ldg(A_ptr + 1);
        float4 B_data1 = __ldg(B_ptr + 1);
        uint32_t sfa_vec = __ldg(reinterpret_cast<const uint32_t*>(SFA_row + (k_offset >> 4)));
        uint32_t sfb_vec = __ldg(reinterpret_cast<const uint32_t*>(SFB_base + (k_offset >> 4)));

        const uint32_t* A0_u32 = reinterpret_cast<const uint32_t*>(&A_data0);
        const uint32_t* A1_u32 = reinterpret_cast<const uint32_t*>(&A_data1);
        const uint32_t* B0_u32 = reinterpret_cast<const uint32_t*>(&B_data0);
        const uint32_t* B1_u32 = reinterpret_cast<const uint32_t*>(&B_data1);

        process_tile_fused(
            local_sum,
            A0_u32[0], A0_u32[1], A0_u32[2], A0_u32[3],
            A1_u32[0], A1_u32[1], A1_u32[2], A1_u32[3],
            B0_u32[0], B0_u32[1], B0_u32[2], B0_u32[3],
            B1_u32[0], B1_u32[1], B1_u32[2], B1_u32[3],
            sfa_vec, sfb_vec);
    }

    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffff, local_sum, offset, 16);
    }

    if (tidx == 0 && valid_row) {
        C[batch_idx * M + global_row] = __float2half(local_sum);
    }
}

torch::Tensor gemv_fp4_fp8_fp16(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C,
    int M, int K, int L)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(SFA.is_cuda(), "SFA must be a CUDA tensor");
    TORCH_CHECK(SFB.is_cuda(), "SFB must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");

    constexpr int ROWS_PER_BLOCK = 8;
    constexpr int THREADS_PER_ROW = 16;

    const dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, L);
    const dim3 block(THREADS_PER_ROW, ROWS_PER_BLOCK, 1);

    block_scaled_gemv_fp4_fp8_fp16<<<grid, block>>>(
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        reinterpret_cast<const uint8_t*>(SFA.data_ptr()),
        reinterpret_cast<const uint8_t*>(SFB.data_ptr()),
        reinterpret_cast<__half*>(C.data_ptr()),
        M, K, L);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return C;
}
"""

gemv_cpp_src = """
#include <torch/extension.h>

torch::Tensor gemv_fp4_fp8_fp16(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C,
    int M, int K, int L);
"""

_gemm_module = load_inline(
    name="block_scaled_gemv_aggressive_fuse_v2",
    cpp_sources=gemv_cpp_src,
    cuda_sources=gemv_cuda_src,
    functions=["gemv_fp4_fp8_fp16"],
    extra_cuda_cflags=[
        # '-O3',
        # '--use_fast_math',
        # '-std=c++17',
        # '--expt-relaxed-constexpr',
        # '--maxrregcount=255',
        # '--prec-div=false', 
        # '--fmad=true',
        # '--ftz=true',
        # '-lineinfo',
        '-gencode=arch=compute_100a,code=sm_100a',
    ],
    verbose=True,
)

def custom_kernel(data):
    a, b, sfa, sfb, _, _, c = data
    m, k_packed, l = a.shape
    k = k_packed * 2

    a_uint8 = a.view(torch.uint8)
    b_uint8 = b.view(torch.uint8)
    sfa_uint8 = sfa.view(torch.uint8)
    sfb_uint8 = sfb.view(torch.uint8)

    _gemm_module.gemv_fp4_fp8_fp16(a_uint8, b_uint8, sfa_uint8, sfb_uint8, c, m, k, l)

    return c
