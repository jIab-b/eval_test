import sys
import os

# Fix for multiprocessing subprocess where sys.stdout/stderr may be None
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

gemv_cuda_source = r"""
// NVFP4 GEMV Kernel
// Operation: c[m,l] = sum_k( a[m,k,l] * sfa[m,k/16,l] * b[0,k,l] * sfb[0,k/16,l] )
//
// Tensor layouts (all CUDA):
//   a:   [M, K/2, L]   float4_e2m1fn_x2  stride=(K/2, 1, M*K/2)    K contiguous
//   b:   [128, K/2, L] float4_e2m1fn_x2  stride=(K/2, 1, 128*K/2)  K contiguous, only row 0 used
//   sfa: [M, K/16, L]  float8_e4m3fn     stride=(K/16, 1, M*K/16)  K contiguous
//   sfb: [128, K/16, L] float8_e4m3fn    stride=(K/16, 1, 128*K/16) K contiguous, only row 0 used
//   c:   [M, 1, L]     float16           stride=(1, 1, M)          M contiguous
//
// Memory layout: For a/b/sfa/sfb, K dimension is innermost (stride=1).
//                For c, M dimension is innermost (stride=1).
// Indexing: a[m,k,l] = a_ptr[m * (K/2) + k + l * (M*K/2)]
//           b[n,k,l] = b_ptr[n * (K/2) + k + l * (128*K/2)]  (use n=0)
//           c[m,l]   = c_ptr[m + l * M]

// 256-bit load using longlong4_32a (32-byte aligned, generates single ld.global.v4.b64)
__device__ __forceinline__ void load_2atoms(const uint4* ptr, uint4& atom0, uint4& atom1) {
    const longlong4_32a* ptr256 = reinterpret_cast<const longlong4_32a*>(ptr);
    longlong4_32a data = *ptr256;
    atom0 = *reinterpret_cast<uint4*>(&data.x);
    atom1 = *reinterpret_cast<uint4*>(&data.z);
}

// Dot product of 32 FP4 values (16 bytes each from A and B) with block scaling
// Each atom has 2 scale blocks of 16 elements, so scales are packed as 2xFP8 in uint16_t
// Uses interleaved A/B input layout for better register allocation
__device__ __forceinline__ __half blockscaled_dot32(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,  // 16 bytes = 32 FP4 from A
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,  // 16 bytes = 32 FP4 from B
    uint16_t scales_a_packed,  // 2 FP8 E4M3 scales for A (lo=first 16, hi=second 16)
    uint16_t scales_b_packed)  // 2 FP8 E4M3 scales for B
{
    __half result;
    asm volatile(
        "{\n"
        // Each group holds 4 bytes from A (0-3) and 4 bytes from B (4-7)
        // Group 0: a0/b0 (elements 0-7), Group 1: a1/b1 (elements 8-15)
        // Group 2: a2/b2 (elements 16-23), Group 3: a3/b3 (elements 24-31)
        ".reg .b8 g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7;\n"
        ".reg .b8 g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7;\n"
        ".reg .b8 g2_0, g2_1, g2_2, g2_3, g2_4, g2_5, g2_6, g2_7;\n"
        ".reg .b8 g3_0, g3_1, g3_2, g3_3, g3_4, g3_5, g3_6, g3_7;\n"
        // Converted f16x2: gN_a_M = A data, gN_b_M = B data
        ".reg .f16x2 g0_a_0, g0_a_1, g0_a_2, g0_a_3, g0_b_0, g0_b_1, g0_b_2, g0_b_3;\n"
        ".reg .f16x2 g1_a_0, g1_a_1, g1_a_2, g1_a_3, g1_b_0, g1_b_1, g1_b_2, g1_b_3;\n"
        ".reg .f16x2 g2_a_0, g2_a_1, g2_a_2, g2_a_3, g2_b_0, g2_b_1, g2_b_2, g2_b_3;\n"
        ".reg .f16x2 g3_a_0, g3_a_1, g3_a_2, g3_a_3, g3_b_0, g3_b_1, g3_b_2, g3_b_3;\n"
        // Partial sums per group (4 accumulators each)
        ".reg .f16x2 s0_0, s0_1, s0_2, s0_3;\n"
        ".reg .f16x2 s1_0, s1_1, s1_2, s1_3;\n"
        ".reg .f16x2 s2_0, s2_1, s2_2, s2_3;\n"
        ".reg .f16x2 s3_0, s3_1, s3_2, s3_3;\n"
        // Scales
        ".reg .f16x2 sfa_f16x2, sfb_f16x2, sf_combined;\n"
        ".reg .f16 sf_lo, sf_hi, lo, hi;\n"
        ".reg .f16x2 sf_lo_x2, sf_hi_x2;\n"

        // Convert packed FP8 scales to FP16x2, multiply, broadcast
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\n"
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %2;\n"
        "mul.rn.f16x2 sf_combined, sfa_f16x2, sfb_f16x2;\n"
        "mov.b32 {sf_lo, sf_hi}, sf_combined;\n"
        "mov.b32 sf_lo_x2, {sf_lo, sf_lo};\n"
        "mov.b32 sf_hi_x2, {sf_hi, sf_hi};\n"

        // Zero accumulators
        "mov.b32 s0_0, 0; mov.b32 s0_1, 0; mov.b32 s0_2, 0; mov.b32 s0_3, 0;\n"
        "mov.b32 s1_0, 0; mov.b32 s1_1, 0; mov.b32 s1_2, 0; mov.b32 s1_3, 0;\n"
        "mov.b32 s2_0, 0; mov.b32 s2_1, 0; mov.b32 s2_2, 0; mov.b32 s2_3, 0;\n"
        "mov.b32 s3_0, 0; mov.b32 s3_1, 0; mov.b32 s3_2, 0; mov.b32 s3_3, 0;\n"

        // Unpack interleaved: each group gets A bytes (0-3) and B bytes (4-7)
        "mov.b32 {g0_0, g0_1, g0_2, g0_3}, %3;\n"   // a0
        "mov.b32 {g0_4, g0_5, g0_6, g0_7}, %4;\n"   // b0
        "mov.b32 {g1_0, g1_1, g1_2, g1_3}, %5;\n"   // a1
        "mov.b32 {g1_4, g1_5, g1_6, g1_7}, %6;\n"   // b1
        "mov.b32 {g2_0, g2_1, g2_2, g2_3}, %7;\n"   // a2
        "mov.b32 {g2_4, g2_5, g2_6, g2_7}, %8;\n"   // b2
        "mov.b32 {g3_0, g3_1, g3_2, g3_3}, %9;\n"   // a3
        "mov.b32 {g3_4, g3_5, g3_6, g3_7}, %10;\n"  // b3

        // Convert all bytes to f16x2 (FP4x2 -> FP16x2)
        "cvt.rn.f16x2.e2m1x2 g0_a_0, g0_0; cvt.rn.f16x2.e2m1x2 g0_a_1, g0_1;\n"
        "cvt.rn.f16x2.e2m1x2 g0_a_2, g0_2; cvt.rn.f16x2.e2m1x2 g0_a_3, g0_3;\n"
        "cvt.rn.f16x2.e2m1x2 g0_b_0, g0_4; cvt.rn.f16x2.e2m1x2 g0_b_1, g0_5;\n"
        "cvt.rn.f16x2.e2m1x2 g0_b_2, g0_6; cvt.rn.f16x2.e2m1x2 g0_b_3, g0_7;\n"

        "cvt.rn.f16x2.e2m1x2 g1_a_0, g1_0; cvt.rn.f16x2.e2m1x2 g1_a_1, g1_1;\n"
        "cvt.rn.f16x2.e2m1x2 g1_a_2, g1_2; cvt.rn.f16x2.e2m1x2 g1_a_3, g1_3;\n"
        "cvt.rn.f16x2.e2m1x2 g1_b_0, g1_4; cvt.rn.f16x2.e2m1x2 g1_b_1, g1_5;\n"
        "cvt.rn.f16x2.e2m1x2 g1_b_2, g1_6; cvt.rn.f16x2.e2m1x2 g1_b_3, g1_7;\n"

        "cvt.rn.f16x2.e2m1x2 g2_a_0, g2_0; cvt.rn.f16x2.e2m1x2 g2_a_1, g2_1;\n"
        "cvt.rn.f16x2.e2m1x2 g2_a_2, g2_2; cvt.rn.f16x2.e2m1x2 g2_a_3, g2_3;\n"
        "cvt.rn.f16x2.e2m1x2 g2_b_0, g2_4; cvt.rn.f16x2.e2m1x2 g2_b_1, g2_5;\n"
        "cvt.rn.f16x2.e2m1x2 g2_b_2, g2_6; cvt.rn.f16x2.e2m1x2 g2_b_3, g2_7;\n"

        "cvt.rn.f16x2.e2m1x2 g3_a_0, g3_0; cvt.rn.f16x2.e2m1x2 g3_a_1, g3_1;\n"
        "cvt.rn.f16x2.e2m1x2 g3_a_2, g3_2; cvt.rn.f16x2.e2m1x2 g3_a_3, g3_3;\n"
        "cvt.rn.f16x2.e2m1x2 g3_b_0, g3_4; cvt.rn.f16x2.e2m1x2 g3_b_1, g3_5;\n"
        "cvt.rn.f16x2.e2m1x2 g3_b_2, g3_6; cvt.rn.f16x2.e2m1x2 g3_b_3, g3_7;\n"

        // FMA: each group computes A[i] * B[i] for 8 elements
        "fma.rn.f16x2 s0_0, g0_a_0, g0_b_0, s0_0; fma.rn.f16x2 s0_1, g0_a_1, g0_b_1, s0_1;\n"
        "fma.rn.f16x2 s0_2, g0_a_2, g0_b_2, s0_2; fma.rn.f16x2 s0_3, g0_a_3, g0_b_3, s0_3;\n"

        "fma.rn.f16x2 s1_0, g1_a_0, g1_b_0, s1_0; fma.rn.f16x2 s1_1, g1_a_1, g1_b_1, s1_1;\n"
        "fma.rn.f16x2 s1_2, g1_a_2, g1_b_2, s1_2; fma.rn.f16x2 s1_3, g1_a_3, g1_b_3, s1_3;\n"

        "fma.rn.f16x2 s2_0, g2_a_0, g2_b_0, s2_0; fma.rn.f16x2 s2_1, g2_a_1, g2_b_1, s2_1;\n"
        "fma.rn.f16x2 s2_2, g2_a_2, g2_b_2, s2_2; fma.rn.f16x2 s2_3, g2_a_3, g2_b_3, s2_3;\n"

        "fma.rn.f16x2 s3_0, g3_a_0, g3_b_0, s3_0; fma.rn.f16x2 s3_1, g3_a_1, g3_b_1, s3_1;\n"
        "fma.rn.f16x2 s3_2, g3_a_2, g3_b_2, s3_2; fma.rn.f16x2 s3_3, g3_a_3, g3_b_3, s3_3;\n"

        // Reduce within each group: 4 -> 2 -> 1
        "add.rn.f16x2 s0_0, s0_0, s0_1; add.rn.f16x2 s0_2, s0_2, s0_3;\n"
        "add.rn.f16x2 s1_0, s1_0, s1_1; add.rn.f16x2 s1_2, s1_2, s1_3;\n"
        "add.rn.f16x2 s2_0, s2_0, s2_1; add.rn.f16x2 s2_2, s2_2, s2_3;\n"
        "add.rn.f16x2 s3_0, s3_0, s3_1; add.rn.f16x2 s3_2, s3_2, s3_3;\n"

        "add.rn.f16x2 s0_0, s0_0, s0_2;\n"
        "add.rn.f16x2 s1_0, s1_0, s1_2;\n"
        "add.rn.f16x2 s2_0, s2_0, s2_2;\n"
        "add.rn.f16x2 s3_0, s3_0, s3_2;\n"

        // Merge groups: 0+1 = first scale block (elements 0-15), 2+3 = second (16-31)
        "add.rn.f16x2 s0_0, s0_0, s1_0;\n"
        "add.rn.f16x2 s2_0, s2_0, s3_0;\n"

        // Apply respective scales
        "mul.rn.f16x2 s0_0, sf_lo_x2, s0_0;\n"
        "mul.rn.f16x2 s2_0, sf_hi_x2, s2_0;\n"

        // Final merge and horizontal reduction
        "add.rn.f16x2 s0_0, s0_0, s2_0;\n"
        "mov.b32 {lo, hi}, s0_0;\n"
        "add.rn.f16 lo, lo, hi;\n"
        "mov.b16 %0, lo;\n"
        "}\n"
        : "=h"(*reinterpret_cast<uint16_t*>(&result))
        : "h"(scales_a_packed), "h"(scales_b_packed),
          "r"(a0), "r"(b0), "r"(a1), "r"(b1),
          "r"(a2), "r"(b2), "r"(a3), "r"(b3)
    );
    return result;
}

// =============================================================================
// Pipelined Kernel for Problem 2 (M=4096, K=7168, L=8)
// Uses cp.async with shared memory for B vector reuse across rows
// =============================================================================

// Thread configuration for pipelined kernel
static constexpr int kThreadsPerRow = 16;    // K-dimension threads
static constexpr int kThreadsPerCol = 8;     // M-dimension threads (rows per block)
static constexpr int kPipeThreadCount = 128; // Total threads per block
static constexpr int kElementsPerAccess = 64; // FP4 elements per thread per K-tile (2x 128-bit loads)
static constexpr int kTileK = kThreadsPerRow * kElementsPerAccess;  // 1024 FP4 per K-tile

// Shared memory sizes per stage (for 256-bit effective loads)
static constexpr int kSmemAPerStage = kPipeThreadCount * 32;      // 128 threads * 32 bytes = 4096
static constexpr int kSmemBPerStage = kThreadsPerRow * 32;        // 16 threads * 32 bytes = 512
static constexpr int kSmemSFAPerStage = kPipeThreadCount * 4;     // 128 threads * 4 bytes = 512
static constexpr int kSmemSFBPerStage = kThreadsPerRow * 4;       // 16 threads * 4 bytes = 64

// Templated shared storage for pipelined kernel
template<int STAGES, int BUFFERS>
struct PipeSharedStorage {
    alignas(16) uint8_t smem_A[BUFFERS][STAGES][kSmemAPerStage];
    alignas(16) uint8_t smem_B[BUFFERS][STAGES][kSmemBPerStage];
    alignas(16) uint8_t smem_SFA[BUFFERS][STAGES][kSmemSFAPerStage];
    alignas(16) uint8_t smem_SFB[BUFFERS][STAGES][kSmemSFBPerStage];
};

// Get shared memory address for cp.async
__device__ __forceinline__ uint32_t smem_addr(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Load a single stage from global to shared memory
// Each thread issues TWO 16-byte loads for 256-bit effective bandwidth
template<int STAGES, int BUFFERS>
__device__ __forceinline__ void load_stage_async(
    PipeSharedStorage<STAGES, BUFFERS>& smem,
    int buffer_idx,
    int stage_idx,
    const uint8_t* __restrict__ ptr_A,
    const uint8_t* __restrict__ ptr_B,
    const uint8_t* __restrict__ ptr_SFA,
    const uint8_t* __restrict__ ptr_SFB,
    int k_byte_offset,
    int sf_byte_offset,
    bool valid)
{
    const int tid = threadIdx.y * kThreadsPerRow + threadIdx.x;
    const bool load_b = (threadIdx.y == 0);

    // A: each thread loads 32 bytes (64 FP4) via two 16-byte loads
    uint32_t smem_a0 = smem_addr(&smem.smem_A[buffer_idx][stage_idx][tid * 32]);
    uint32_t smem_a1 = smem_addr(&smem.smem_A[buffer_idx][stage_idx][tid * 32 + 16]);

    if (valid) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.cg.shared.global [%2], [%3], 16;\n"
            :: "r"(smem_a0), "l"(ptr_A + k_byte_offset + threadIdx.x * 32),
               "r"(smem_a1), "l"(ptr_A + k_byte_offset + threadIdx.x * 32 + 16)
        );
    } else {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
            "cp.async.cg.shared.global [%2], [%3], 16, 0;\n"
            :: "r"(smem_a0), "l"(ptr_A + k_byte_offset + threadIdx.x * 32),
               "r"(smem_a1), "l"(ptr_A + k_byte_offset + threadIdx.x * 32 + 16)
        );
    }

    // B: only threadIdx.y == 0 loads (shared across all 8 rows) - also 32 bytes
    if (load_b) {
        uint32_t smem_b0 = smem_addr(&smem.smem_B[buffer_idx][stage_idx][threadIdx.x * 32]);
        uint32_t smem_b1 = smem_addr(&smem.smem_B[buffer_idx][stage_idx][threadIdx.x * 32 + 16]);
        if (valid) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                "cp.async.cg.shared.global [%2], [%3], 16;\n"
                :: "r"(smem_b0), "l"(ptr_B + k_byte_offset + threadIdx.x * 32),
                   "r"(smem_b1), "l"(ptr_B + k_byte_offset + threadIdx.x * 32 + 16)
            );
        } else {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                "cp.async.cg.shared.global [%2], [%3], 16, 0;\n"
                :: "r"(smem_b0), "l"(ptr_B + k_byte_offset + threadIdx.x * 32),
                   "r"(smem_b1), "l"(ptr_B + k_byte_offset + threadIdx.x * 32 + 16)
            );
        }
    }

    // SFA: each thread loads 4 bytes (4 FP8 scales for 64 FP4 elements)
    uint32_t smem_sfa = smem_addr(&smem.smem_SFA[buffer_idx][stage_idx][tid * 4]);
    if (valid) {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;\n"
            :: "r"(smem_sfa), "l"(ptr_SFA + sf_byte_offset + threadIdx.x * 4)
        );
    } else {
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4, 0;\n"
            :: "r"(smem_sfa), "l"(ptr_SFA + sf_byte_offset + threadIdx.x * 4)
        );
    }

    // SFB: only row 0 loads (4 bytes for 64 FP4 elements)
    if (load_b) {
        uint32_t smem_sfb = smem_addr(&smem.smem_SFB[buffer_idx][stage_idx][threadIdx.x * 4]);
        if (valid) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(smem_sfb), "l"(ptr_SFB + sf_byte_offset + threadIdx.x * 4)
            );
        } else {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4, 0;\n"
                :: "r"(smem_sfb), "l"(ptr_SFB + sf_byte_offset + threadIdx.x * 4)
            );
        }
    }
}

// Pipelined kernel with cp.async and shared memory B reuse
template<int M, int K, int L, int STAGES = 4, int BUFFERS = 1>
__global__ void __launch_bounds__(128)
nvfp4_gemv_pipelined(
    const uint8_t* __restrict__ A,      // [L, M, K/2] packed FP4
    const uint8_t* __restrict__ scale_A, // [L, M, K/16] FP8 scales
    const uint8_t* __restrict__ B,      // [L, 128, K/2] packed FP4 (padded)
    const uint8_t* __restrict__ scale_B, // [L, 128, K/16] FP8 scales
    __half* __restrict__ output)         // [L, M]
{
    constexpr int kTotalTiles = K / kTileK;
    constexpr int kBytesPerTileA = kTileK / 2;   // 512 bytes per K-tile
    constexpr int kSFBytesPerTile = kTileK / 16; // 64 bytes per K-tile
    constexpr int kRowStrideA = K / 2;           // Bytes per row in A
    constexpr int kRowStrideSFA = K / 16;        // Bytes per row in scale_A

    __shared__ PipeSharedStorage<STAGES, BUFFERS> smem;

    const int row_in_block = threadIdx.y;
    const int k_lane = threadIdx.x;
    const int block_row_base = blockIdx.x * kThreadsPerCol;
    const int batch = blockIdx.z;
    const int global_row = block_row_base + row_in_block;

    if (global_row >= M) return;

    // Setup base pointers - B has 128 rows padding per batch
    const uint8_t* ptr_A_base = A + (size_t)batch * M * kRowStrideA + (size_t)global_row * kRowStrideA;
    const uint8_t* ptr_B_base = B + (size_t)batch * 128 * kRowStrideA;  // 128 rows padding
    const uint8_t* ptr_SFA_base = scale_A + (size_t)batch * M * kRowStrideSFA + (size_t)global_row * kRowStrideSFA;
    const uint8_t* ptr_SFB_base = scale_B + (size_t)batch * 128 * kRowStrideSFA;

    float accum = 0.0f;

    // === PROLOGUE: Prime buffer 0 with STAGES tiles ===
    #pragma unroll
    for (int s = 0; s < STAGES; s++) {
        int k_tile = s;
        bool valid = (k_tile < kTotalTiles);
        load_stage_async(smem, 0, s,
                        ptr_A_base, ptr_B_base, ptr_SFA_base, ptr_SFB_base,
                        k_tile * kBytesPerTileA,
                        k_tile * kSFBytesPerTile,
                        valid);
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    int smem_read = 0;
    int smem_write = (BUFFERS > 1) ? 1 : 0;
    int tile_idx = 0;

    // === MAINLOOP ===
    while (tile_idx < kTotalTiles) {
        int smem_read_curr = smem_read;

        #pragma unroll
        for (int stage = 0; stage < STAGES; stage++) {
            int current_tile = tile_idx + stage;
            if (current_tile >= kTotalTiles) break;

            // At first stage: issue async loads for next buffer
            if (stage == 0) {
                int next_tile_base = tile_idx + STAGES;
                if (next_tile_base < kTotalTiles) {
                    #pragma unroll
                    for (int s = 0; s < STAGES; s++) {
                        int k_tile = next_tile_base + s;
                        bool valid = (k_tile < kTotalTiles);
                        load_stage_async(smem, smem_write, s,
                                        ptr_A_base, ptr_B_base, ptr_SFA_base, ptr_SFB_base,
                                        k_tile * kBytesPerTileA,
                                        k_tile * kSFBytesPerTile,
                                        valid);
                    }
                    asm volatile("cp.async.commit_group;\n");

                    if constexpr (BUFFERS > 1) {
                        int tmp = smem_write;
                        smem_write = smem_read;
                        smem_read = tmp;
                    }
                }
            }

            // Wait for data at last stage
            if constexpr (BUFFERS > 1) {
                if (stage == STAGES - 1) {
                    asm volatile("cp.async.wait_group 0;\n");
                    __syncthreads();
                }
            }

            // Load from shared memory and compute
            const int tid = threadIdx.y * kThreadsPerRow + threadIdx.x;

            uint4 frag_a0 = *reinterpret_cast<const uint4*>(&smem.smem_A[smem_read_curr][stage][tid * 32]);
            uint4 frag_a1 = *reinterpret_cast<const uint4*>(&smem.smem_A[smem_read_curr][stage][tid * 32 + 16]);
            uint4 frag_b0 = *reinterpret_cast<const uint4*>(&smem.smem_B[smem_read_curr][stage][k_lane * 32]);
            uint4 frag_b1 = *reinterpret_cast<const uint4*>(&smem.smem_B[smem_read_curr][stage][k_lane * 32 + 16]);

            uint32_t sfa_packed = *reinterpret_cast<const uint32_t*>(&smem.smem_SFA[smem_read_curr][stage][tid * 4]);
            uint32_t sfb_packed = *reinterpret_cast<const uint32_t*>(&smem.smem_SFB[smem_read_curr][stage][k_lane * 4]);

            uint16_t sfa0 = sfa_packed & 0xFFFF;
            uint16_t sfa1 = sfa_packed >> 16;
            uint16_t sfb0 = sfb_packed & 0xFFFF;
            uint16_t sfb1 = sfb_packed >> 16;

            __half dot0 = blockscaled_dot32(
                frag_a0.x, frag_a0.y, frag_a0.z, frag_a0.w,
                frag_b0.x, frag_b0.y, frag_b0.z, frag_b0.w,
                sfa0, sfb0
            );
            __half dot1 = blockscaled_dot32(
                frag_a1.x, frag_a1.y, frag_a1.z, frag_a1.w,
                frag_b1.x, frag_b1.y, frag_b1.z, frag_b1.w,
                sfa1, sfb1
            );
            accum += __half2float(dot0) + __half2float(dot1);
        }
        tile_idx += STAGES;
    }

    // Drain remaining async copies
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // === REDUCTION across K-threads (16 threads) ===
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        accum += __shfl_xor_sync(0xFFFF, accum, mask, 16);
    }

    // Write output
    if (k_lane == 0) {
        output[batch * M + global_row] = __float2half(accum);
    }
}

// =============================================================================
// Generic Runtime Kernel (any size, dimensions divisible by 64)
// =============================================================================

__global__ void nvfp4_gemv_generic(
    const uint4* __restrict__ A,
    const uint16_t* __restrict__ scale_A,
    const uint4* __restrict__ B,
    const uint16_t* __restrict__ scale_B,
    __half* __restrict__ output,
    int M, int K, int L)
{
    const int atoms_per_row = K / 32;
    const int total_rows = M * L;

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;

    int global_row = blockIdx.x * warps_per_block + warp_id;
    if (global_row >= total_rows) return;

    // Batch-interleaved ordering (matches template)
    int row = global_row / L;
    int batch = global_row % L;

    int a_base = batch * M * atoms_per_row + row * atoms_per_row;
    // B has 128 rows padding per batch (only row 0 is used)
    int b_base = batch * 128 * atoms_per_row;

    float accumulator = 0.0f;

    // Process 64 atoms per iteration (32 lanes x 2 atoms via 256-bit loads)
    int full_iters = atoms_per_row / 64;
    int remainder_atoms = atoms_per_row % 64;

    for (int iter = 0; iter < full_iters; iter++) {
        int atom_idx = iter * 64 + lane_id * 2;

        uint4 a0, a1;
        load_2atoms(&A[a_base + atom_idx], a0, a1);
        uint32_t scales_a = *reinterpret_cast<const uint32_t*>(&scale_A[a_base + atom_idx]);

        uint4 b0, b1;
        load_2atoms(&B[b_base + atom_idx], b0, b1);
        uint32_t scales_b = *reinterpret_cast<const uint32_t*>(&scale_B[b_base + atom_idx]);

        __half dot0 = blockscaled_dot32(a0.x, a0.y, a0.z, a0.w, b0.x, b0.y, b0.z, b0.w,
            (uint16_t)(scales_a & 0xFFFF), (uint16_t)(scales_b & 0xFFFF));
        __half dot1 = blockscaled_dot32(a1.x, a1.y, a1.z, a1.w, b1.x, b1.y, b1.z, b1.w,
            (uint16_t)(scales_a >> 16), (uint16_t)(scales_b >> 16));

        accumulator += __half2float(__hadd(dot0, dot1));
    }

    // Remainder atoms (0 to 62, always even since K % 64 == 0)
    if (remainder_atoms > 0 && lane_id * 2 < remainder_atoms) {
        int atom_idx = full_iters * 64 + lane_id * 2;

        uint4 a0, a1;
        load_2atoms(&A[a_base + atom_idx], a0, a1);
        uint32_t scales_a = *reinterpret_cast<const uint32_t*>(&scale_A[a_base + atom_idx]);

        uint4 b0, b1;
        load_2atoms(&B[b_base + atom_idx], b0, b1);
        uint32_t scales_b = *reinterpret_cast<const uint32_t*>(&scale_B[b_base + atom_idx]);

        __half dot0 = blockscaled_dot32(a0.x, a0.y, a0.z, a0.w, b0.x, b0.y, b0.z, b0.w,
            (uint16_t)(scales_a & 0xFFFF), (uint16_t)(scales_b & 0xFFFF));
        __half dot1 = blockscaled_dot32(a1.x, a1.y, a1.z, a1.w, b1.x, b1.y, b1.z, b1.w,
            (uint16_t)(scales_a >> 16), (uint16_t)(scales_b >> 16));

        accumulator += __half2float(__hadd(dot0, dot1));
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xFFFFFFFF, accumulator, offset);
    }

    if (lane_id == 0) {
        // PyTorch output layout: c[m,l] = ptr[m + l*M] (M contiguous, L outermost)
        output[batch * M + row] = __float2half(accumulator);
    }
}

// =============================================================================
// Template Kernel (compile-time optimized)
// =============================================================================

template<int M, int K, int L, int TPB, int WARPS_PER_ROW>
__global__ void nvfp4_gemv_template(
    const uint4* __restrict__ A,
    const uint16_t* __restrict__ scale_A,
    const uint4* __restrict__ B,
    const uint16_t* __restrict__ scale_B,
    __half* __restrict__ output)
{
    constexpr int ATOMS_PER_ROW = K / 32;
    constexpr int WARPS_PER_BLOCK = TPB / 32;
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK / WARPS_PER_ROW;
    constexpr int ATOMS_PER_LANE = ATOMS_PER_ROW / (32 * WARPS_PER_ROW);
    constexpr int FULL_ITERS = ATOMS_PER_LANE / 2;
    constexpr int REMAINDER = ATOMS_PER_LANE % 2;
    constexpr bool NEED_CROSS_WARP = (WARPS_PER_ROW > 1);

    // Need separate slots for each row in block when doing cross-warp reduction
    __shared__ float warp_partial_sums[NEED_CROSS_WARP ? ROWS_PER_BLOCK * WARPS_PER_ROW : 1];

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    // Row assignment
    int row_in_block = warp_id / WARPS_PER_ROW;
    int warp_in_row = warp_id % WARPS_PER_ROW;
    int global_row = blockIdx.x * ROWS_PER_BLOCK + row_in_block;

    if (global_row >= M * L) return;

    // Batch/row decomposition (for L > 1)
    // Original ordering: batch varies fastest (interleaved)
    int batch, row;
    if constexpr (L == 1) {
        batch = 0;
        row = global_row;
    } else {
        row = global_row / L;
        batch = global_row % L;
    }

    // Base indices
    int a_base = batch * M * ATOMS_PER_ROW + row * ATOMS_PER_ROW;
    // B has 128 rows padding per batch (only row 0 is used)
    int b_base = batch * 128 * ATOMS_PER_ROW;
    int lane_offset = warp_in_row * 32 + lane_id;

    float accumulator = 0.0f;

    // Main loop: 2 atoms per iteration (256-bit loads)
    if constexpr (FULL_ITERS > 0) {
        #pragma unroll
        for (int iter = 0; iter < FULL_ITERS; iter++) {
            int atom_idx = iter * (32 * WARPS_PER_ROW * 2) + lane_offset * 2;

            uint4 a0, a1;
            load_2atoms(&A[a_base + atom_idx], a0, a1);
            uint32_t scales_a = *reinterpret_cast<const uint32_t*>(&scale_A[a_base + atom_idx]);

            uint4 b0, b1;
            load_2atoms(&B[b_base + atom_idx], b0, b1);
            uint32_t scales_b = *reinterpret_cast<const uint32_t*>(&scale_B[b_base + atom_idx]);

            __half dot0 = blockscaled_dot32(a0.x, a0.y, a0.z, a0.w, b0.x, b0.y, b0.z, b0.w,
                (uint16_t)(scales_a & 0xFFFF), (uint16_t)(scales_b & 0xFFFF));
            __half dot1 = blockscaled_dot32(a1.x, a1.y, a1.z, a1.w, b1.x, b1.y, b1.z, b1.w,
                (uint16_t)(scales_a >> 16), (uint16_t)(scales_b >> 16));

            accumulator += __half2float(__hadd(dot0, dot1));
        }
    }

    // Remainder: single atom when ATOMS_PER_LANE is odd
    if constexpr (REMAINDER == 1) {
        int atom_idx = FULL_ITERS * (32 * WARPS_PER_ROW * 2) + lane_offset;

        uint4 a0 = A[a_base + atom_idx];
        uint16_t scales_a = scale_A[a_base + atom_idx];
        uint4 b0 = B[b_base + atom_idx];
        uint16_t scales_b = scale_B[b_base + atom_idx];

        __half dot = blockscaled_dot32(a0.x, a0.y, a0.z, a0.w, b0.x, b0.y, b0.z, b0.w,
            scales_a, scales_b);

        accumulator += __half2float(dot);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_down_sync(0xFFFFFFFF, accumulator, offset);
    }

    // Cross-warp reduction when multiple warps process same row
    if constexpr (NEED_CROSS_WARP) {
        // Each row in block gets its own slots: row_in_block * WARPS_PER_ROW + warp_in_row
        int smem_offset = row_in_block * WARPS_PER_ROW;
        if (lane_id == 0) {
            warp_partial_sums[smem_offset + warp_in_row] = accumulator;
        }
        __syncthreads();

        if (warp_in_row == 0 && lane_id == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_ROW; w++) {
                sum += warp_partial_sums[smem_offset + w];
            }
            // PyTorch output layout: c[m,l] = ptr[m + l*M] (M contiguous, L outermost)
            output[batch * M + row] = __float2half(sum);
        }
    } else {
        if (lane_id == 0) {
            // PyTorch output layout: c[m,l] = ptr[m + l*M] (M contiguous, L outermost)
            output[batch * M + row] = __float2half(accumulator);
        }
    }
}

torch::Tensor gemv_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor c
) {
    // Extract dimensions
    // a: [M, K/2, L] where K/2 is in bytes (float4_e2m1fn_x2 packs 2 FP4 per byte)
    const int M = a.size(0);
    const int K = a.size(1) * 2;  // Convert from packed bytes to actual K
    const int L = a.size(2);

    // Get raw pointers
    // A and B: treat as uint4* (16-byte atoms, each containing 32 FP4 values)
    const uint4* A_ptr = reinterpret_cast<const uint4*>(a.data_ptr());
    const uint4* B_ptr = reinterpret_cast<const uint4*>(b.data_ptr());

    // Scale factors: treat as uint16_t* (2 packed FP8 scales per atom)
    const uint16_t* scale_A_ptr = reinterpret_cast<const uint16_t*>(sfa.data_ptr());
    const uint16_t* scale_B_ptr = reinterpret_cast<const uint16_t*>(sfb.data_ptr());

    // Output: __half*
    __half* output_ptr = reinterpret_cast<__half*>(c.data_ptr());

    // Check for optimized template kernel shapes
    if (M == 7168 && K == 16384 && L == 1) {
        // Problem 1: TPB=128, WARPS_PER_ROW=4 (1 row per block, avoids smem conflicts)
        constexpr int TPB = 128;
        constexpr int WARPS_PER_ROW = 4;
        constexpr int ROWS_PER_BLOCK = (TPB / 32) / WARPS_PER_ROW;
        const int num_blocks = (M * L + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
        nvfp4_gemv_template<7168, 16384, 1, TPB, WARPS_PER_ROW><<<num_blocks, TPB>>>(
            A_ptr, scale_A_ptr, B_ptr, scale_B_ptr, output_ptr
        );
    } else if (M == 4096 && K == 7168 && L == 8) {
        // Problem 2: Pipelined kernel with STAGES=7, BUFFERS=1 (31.4μs)
        // K=7168/1024=7 tiles, S=7 loads all in prologue - no buffer race
        dim3 grid((4096 + 7) / 8, 1, 8);
        dim3 block(kThreadsPerRow, kThreadsPerCol, 1);  // 16 x 8 = 128 threads
        nvfp4_gemv_pipelined<4096, 7168, 8, 7, 1><<<grid, block>>>(
            reinterpret_cast<const uint8_t*>(A_ptr),
            reinterpret_cast<const uint8_t*>(scale_A_ptr),
            reinterpret_cast<const uint8_t*>(B_ptr),
            reinterpret_cast<const uint8_t*>(scale_B_ptr),
            output_ptr
        );
    } else if (M == 7168 && K == 2048 && L == 4) {
        // Problem 3: Pipelined kernel with STAGES=2, BUFFERS=1 (12.1μs - best from sweep)
        // K=2048/1024=2 tiles, S=2 loads all in prologue - no buffer race
        dim3 grid((7168 + 7) / 8, 1, 4);
        dim3 block(kThreadsPerRow, kThreadsPerCol, 1);  // 16 x 8 = 128 threads
        nvfp4_gemv_pipelined<7168, 2048, 4, 2, 1><<<grid, block>>>(
            reinterpret_cast<const uint8_t*>(A_ptr),
            reinterpret_cast<const uint8_t*>(scale_A_ptr),
            reinterpret_cast<const uint8_t*>(B_ptr),
            reinterpret_cast<const uint8_t*>(scale_B_ptr),
            output_ptr
        );
    } else {
        // Generic fallback
        const int TPB = 256;
        const int warps_per_block = TPB / 32;
        const int total_rows = M * L;
        const int num_blocks = (total_rows + warps_per_block - 1) / warps_per_block;
        nvfp4_gemv_generic<<<num_blocks, TPB>>>(
            A_ptr, scale_A_ptr, B_ptr, scale_B_ptr, output_ptr, M, K, L
        );
    }

    return c;
}
"""

gemv_cpp_source = """
#include <torch/extension.h>

torch::Tensor gemv_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfa,
    const torch::Tensor& sfb,
    torch::Tensor c
);
"""

gemv_module = load_inline(
    name='nvfp4_gemv',
    cpp_sources=gemv_cpp_source,
    cuda_sources=gemv_cuda_source,
    functions=['gemv_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo', '-gencode', 'arch=compute_100a,code=sm_100a'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 block-scaled GEMV kernel.

    Args:
        data: Tuple of (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)
            a: [M, K/2, L] - Input matrix in float4_e2m1fn_x2 (CUDA)
            b: [128, K/2, L] - Input vector in float4_e2m1fn_x2 (CUDA, padded, only row 0 used)
            sfa: [M, K/16, L] - Scale factors for A in float8_e4m3fn (CUDA)
            sfb: [128, K/16, L] - Scale factors for B in float8_e4m3fn (CUDA, padded)
            sfa_permuted: ignored
            sfb_permuted: ignored
            c: [M, 1, L] - Output in float16 (CUDA)

    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, sfa, sfb, _, _, c = data

    return gemv_module.gemv_cuda(a, b, sfa, sfb, c)
