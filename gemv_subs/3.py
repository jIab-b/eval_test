
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

static constexpr int kWarpGroupSize  = 32;

__device__ __forceinline__ void decode_fp4_pack_to_half2x4(int *dst_halves, int in) {
    asm volatile(
        "{\n\t"
        ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
        "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
        "}"
        : "=r"(dst_halves[0]), "=r"(dst_halves[1]), "=r"(dst_halves[2]), "=r"(dst_halves[3])
        : "r"(in)
    );
}

__device__ __forceinline__ void decode_fp8_pair_to_half2(int *dst_pair, int16_t in) {
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(dst_pair[0]) : "h"(in));
}

template <int WIDTH>
__device__ __forceinline__ float warp_group_reduce_sum(float lane_accum) {
    if constexpr (WIDTH > 1) {
        #pragma unroll
        for (int shuffle_delta = WIDTH / 2; shuffle_delta > 0; shuffle_delta /= 2) {
            lane_accum += __shfl_down_sync(0xffffffffu, lane_accum, shuffle_delta, WIDTH);
        }
    }
    return lane_accum;
}

__device__ __forceinline__ void load_fp4_noalloc_u64x2(uint64_t (&dst)[2], uint64_t addr) {
    asm volatile(
        "ld.global.nc.L1::no_allocate.v2.u64 {%0, %1}, [%2];\n\t"
        : "=l"(dst[0]), "=l"(dst[1])
        : "l"(addr)
    );
}

__device__ __forceinline__ void load_fp4_cache_u64x2(uint64_t (&dst)[2], uint64_t addr) {
    asm volatile(
        "ld.global.nc.L1::evict_last.v2.u64 {%0, %1}, [%2];\n\t"
        : "=l"(dst[0]), "=l"(dst[1])
        : "l"(addr)
    );
}

__device__ __forceinline__ void load_scale_noalloc_u16(uint16_t &dst, uint64_t addr) {
    asm volatile(
        "ld.global.nc.L1::no_allocate.u16 %0, [%1];\n\t"
        : "=h"(dst)
        : "l"(addr)
    );
}

__device__ __forceinline__ void load_scale_cache_u16(uint16_t &dst, uint64_t addr) {
    asm volatile(
        "ld.global.nc.L1::evict_last.u16 %0, [%1];\n\t"
        : "=h"(dst)
        : "l"(addr)
    );
}

__device__ __forceinline__ void fetch_fp4_operand_group(
    const __nv_fp4x2_e2m1* ptr_A_row,
    const __nv_fp4x2_e2m1* ptr_B_panel,
    int                    block_tile_offset,
    uint64_t (&matrix_frag)[2],
    uint64_t (&vector_frag)[2])
{
    int tile_elem_base = block_tile_offset * 16;
    uint64_t ptr_A_row_addr = reinterpret_cast<uint64_t>(ptr_A_row + tile_elem_base);
    uint64_t ptr_B_panel_addr = reinterpret_cast<uint64_t>(ptr_B_panel + tile_elem_base);

    load_fp4_noalloc_u64x2(matrix_frag, ptr_A_row_addr);
    load_fp4_cache_u64x2(vector_frag, ptr_B_panel_addr);
}

__device__ __forceinline__ void fetch_blockscale_pair(
    const uint16_t* ptr_SFA_u16,
    const uint16_t* ptr_SFB_u16,
    int             block_tile_offset,
    uint16_t       &sfa_scale_regs,
    uint16_t       &sfb_scale_regs)
{
    uint64_t sfa_addr = reinterpret_cast<uint64_t>(ptr_SFA_u16 + block_tile_offset);
    uint64_t sfb_addr = reinterpret_cast<uint64_t>(ptr_SFB_u16 + block_tile_offset);

    load_scale_noalloc_u16(sfa_scale_regs, sfa_addr);
    load_scale_cache_u16(sfb_scale_regs, sfb_addr);
}

__device__ __forceinline__ float accumulate_blockscaled_fp4(
    const uint64_t (&matrix_frag)[2],
    const uint64_t (&vector_frag)[2],
    uint16_t       sfa_scale_regs,
    uint16_t       sfb_scale_regs)
{
    const int* matrix_frag_packed = reinterpret_cast<const int*>(&matrix_frag[0]);
    const int* vector_frag_packed = reinterpret_cast<const int*>(&vector_frag[0]);

    half2 row_scale_fp16x2;
    half2 vec_scale_fp16x2;
    decode_fp8_pair_to_half2(reinterpret_cast<int*>(&row_scale_fp16x2), static_cast<int16_t>(sfa_scale_regs));
    decode_fp8_pair_to_half2(reinterpret_cast<int*>(&vec_scale_fp16x2), static_cast<int16_t>(sfb_scale_regs));
    half2 fused_scales = __hmul2(row_scale_fp16x2, vec_scale_fp16x2);

    half2 acc_group0 = __float2half2_rn(0.f);
    half2 acc_group1 = __float2half2_rn(0.f);

    #pragma unroll
    for (int pack = 0; pack < 4; ++pack) {
        half2 a_chunk[4];
        half2 b_chunk[4];
        decode_fp4_pack_to_half2x4(reinterpret_cast<int*>(a_chunk), matrix_frag_packed[pack]);
        decode_fp4_pack_to_half2x4(reinterpret_cast<int*>(b_chunk), vector_frag_packed[pack]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int tile_iter_idx = pack * 4 + j;
            if (tile_iter_idx < 8) {
                acc_group0 = __hfma2(a_chunk[j], b_chunk[j], acc_group0);
            } else {
                acc_group1 = __hfma2(a_chunk[j], b_chunk[j], acc_group1);
            }
        }
    }

    __half group0 = __hadd(__low2half(acc_group0), __high2half(acc_group0));
    __half group1 = __hadd(__low2half(acc_group1), __high2half(acc_group1));

    __half scale_half0 = __low2half(fused_scales);
    __half scale_half1 = __high2half(fused_scales);

    float scaled_dot = __half2float(__hmul(group0, scale_half0)) +
                       __half2float(__hmul(group1, scale_half1));
    return scaled_dot;
}

struct BlockscaledParams {
    using stride_index_t = uint64_t;

    int k_extent;
    int runtime_tile_iters;

    void *__restrict__ ptr_A;
    void *__restrict__ ptr_B;
    void *__restrict__ ptr_SFA;
    void *__restrict__ ptr_SFB;
    void *__restrict__ ptr_C;

    stride_index_t batch_stride_A;
    stride_index_t batch_stride_B;
    stride_index_t batch_stride_SFA;
    stride_index_t batch_stride_SFB;
    stride_index_t batch_stride_C;

    stride_index_t stride_A;
    stride_index_t stride_SFA;
};

struct RowPointers {
    const __nv_fp4x2_e2m1* ptr_A_row;
    const __nv_fp4x2_e2m1* ptr_B_panel;
    const uint16_t*        ptr_SFA_u16;
    const uint16_t*        ptr_SFB_u16;
    size_t                 batch_offset_C;
};

__device__ __forceinline__ RowPointers compose_row_pointers(const BlockscaledParams& kernel_params,
                                                         int batch_idx,
                                                         int row_idx)
{
    const size_t batch_offset_A   = static_cast<size_t>(batch_idx) * kernel_params.batch_stride_A;
    const size_t batch_offset_SFA = static_cast<size_t>(batch_idx) * kernel_params.batch_stride_SFA;
    const size_t batch_offset_B   = static_cast<size_t>(batch_idx) * kernel_params.batch_stride_B;
    const size_t batch_offset_SFB = static_cast<size_t>(batch_idx) * kernel_params.batch_stride_SFB;
    const size_t batch_offset_C   = static_cast<size_t>(batch_idx) * kernel_params.batch_stride_C;

    RowPointers row_ptrs{};
    row_ptrs.ptr_A_row = static_cast<const __nv_fp4x2_e2m1*>(kernel_params.ptr_A) + batch_offset_A + row_idx * kernel_params.stride_A;
    const __nv_fp8_e4m3* sfa_vec = static_cast<const __nv_fp8_e4m3*>(kernel_params.ptr_SFA) + batch_offset_SFA + row_idx * kernel_params.stride_SFA;
    row_ptrs.ptr_B_panel = static_cast<const __nv_fp4x2_e2m1*>(kernel_params.ptr_B) + batch_offset_B;
    const __nv_fp8_e4m3* sfb_vec = static_cast<const __nv_fp8_e4m3*>(kernel_params.ptr_SFB) + batch_offset_SFB;
    row_ptrs.ptr_SFA_u16 = reinterpret_cast<const uint16_t*>(sfa_vec);
    row_ptrs.ptr_SFB_u16 = reinterpret_cast<const uint16_t*>(sfb_vec);
    row_ptrs.batch_offset_C = batch_offset_C;
    return row_ptrs;
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS>
__global__ void __launch_bounds__(CTA_ROWS * LANES_PER_STRIPE, 8)
BlockscaledCooperativeKernel(const __grid_constant__ BlockscaledParams kernel_params)
{
    const int thread_linear = threadIdx.x;
    const int row_in_block  = thread_linear / LANES_PER_STRIPE;
    const int lane_idx      = thread_linear % LANES_PER_STRIPE;
    const int batch_idx     = blockIdx.z;
    const int row_idx       = blockIdx.x * CTA_ROWS + row_in_block;

    const RowPointers row_ptrs = compose_row_pointers(kernel_params, batch_idx, row_idx);

    float thread_accum = 0.f;
    #pragma unroll
    for (int tile_iter_idx = 0; tile_iter_idx < STATIC_TILE_ITERS; ++tile_iter_idx) {
        int block_tile_offset = tile_iter_idx * LANES_PER_STRIPE + lane_idx;

        uint64_t matrix_frag[2], vector_frag[2];
        uint16_t sfa_scale_regs, sfb_scale_regs;

        fetch_fp4_operand_group(
            row_ptrs.ptr_A_row, row_ptrs.ptr_B_panel,
            block_tile_offset,
            matrix_frag, vector_frag);
        fetch_blockscale_pair(
            row_ptrs.ptr_SFA_u16, row_ptrs.ptr_SFB_u16,
            block_tile_offset,
            sfa_scale_regs, sfb_scale_regs);
        thread_accum += accumulate_blockscaled_fp4(matrix_frag, vector_frag, sfa_scale_regs, sfb_scale_regs);
    }

    if constexpr (LANES_PER_STRIPE <= kWarpGroupSize) {
        thread_accum = warp_group_reduce_sum<LANES_PER_STRIPE>(thread_accum);
        if (lane_idx == 0) {
            __half* row_output = (__half*)kernel_params.ptr_C + row_ptrs.batch_offset_C + row_idx;
            row_output[0] = __float2half(thread_accum);
        }
    } else {
        static_assert((LANES_PER_STRIPE % kWarpGroupSize) == 0,
                      "LANES_PER_STRIPE must be a multiple of kWarpGroupSize for BlockscaledCooperativeKernel");
        __shared__ float tile_partial_storage[CTA_ROWS * LANES_PER_STRIPE];
        float* row_slice = tile_partial_storage + row_in_block * LANES_PER_STRIPE;
        row_slice[lane_idx] = thread_accum;
        __syncthreads();

        if (lane_idx < kWarpGroupSize) {
            float lane_accum = row_slice[lane_idx];
            #pragma unroll
            for (int shuffle_delta = kWarpGroupSize; shuffle_delta < LANES_PER_STRIPE; shuffle_delta += kWarpGroupSize) {
                lane_accum += row_slice[lane_idx + shuffle_delta];
            }
            lane_accum = warp_group_reduce_sum<kWarpGroupSize>(lane_accum);
            if (lane_idx == 0) {
                __half* row_output = (__half*)kernel_params.ptr_C + row_ptrs.batch_offset_C + row_idx;
                row_output[0] = __float2half(lane_accum);
            }
        }
    }
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS>
__global__ void __launch_bounds__(CTA_ROWS * LANES_PER_STRIPE, 8)
BlockscaledWarpKernel(const __grid_constant__ BlockscaledParams kernel_params)
{
    static_assert(LANES_PER_STRIPE <= kWarpGroupSize,
                  "BlockscaledWarpKernel expects LANES_PER_STRIPE to fit within a single warp");
    const int thread_linear = threadIdx.x;
    const int row_in_block  = thread_linear / LANES_PER_STRIPE;
    const int lane_idx      = thread_linear % LANES_PER_STRIPE;
    const int batch_idx     = blockIdx.z;
    const int row_idx       = blockIdx.x * CTA_ROWS + row_in_block;
    const RowPointers row_ptrs = compose_row_pointers(kernel_params, batch_idx, row_idx);

    float thread_accum = 0.f;

    auto tile_body = [&](int tile_iter_idx) {
        int block_tile_offset = tile_iter_idx * LANES_PER_STRIPE + lane_idx;

        uint64_t matrix_frag[2], vector_frag[2];
        uint16_t sfa_scale_regs, sfb_scale_regs;

        fetch_fp4_operand_group(
            row_ptrs.ptr_A_row, row_ptrs.ptr_B_panel,
            block_tile_offset,
            matrix_frag, vector_frag);
        fetch_blockscale_pair(
            row_ptrs.ptr_SFA_u16, row_ptrs.ptr_SFB_u16,
            block_tile_offset,
            sfa_scale_regs, sfb_scale_regs);
        thread_accum += accumulate_blockscaled_fp4(matrix_frag, vector_frag, sfa_scale_regs, sfb_scale_regs);
    };

    if constexpr (STATIC_TILE_ITERS > 0) {
        #pragma unroll
        for (int tile_iter_idx = 0; tile_iter_idx < STATIC_TILE_ITERS; ++tile_iter_idx) {
            tile_body(tile_iter_idx);
        }
    } else {
        int loop_iters = kernel_params.runtime_tile_iters;
        if (loop_iters == 0) {
            loop_iters = kernel_params.k_extent / (LANES_PER_STRIPE * 16);
        }
        for (int tile_iter_idx = 0; tile_iter_idx < loop_iters; ++tile_iter_idx) {
            tile_body(tile_iter_idx);
        }
    }

    thread_accum = warp_group_reduce_sum<LANES_PER_STRIPE>(thread_accum);

    if (lane_idx == 0) {
        __half* row_output = (__half*)kernel_params.ptr_C + row_ptrs.batch_offset_C + row_idx;
        row_output[0] = __float2half(thread_accum);
    }
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS>
inline void launch_blockscaled_warp(int m_extent, int batch_count, const BlockscaledParams& kernel_params)
{
    dim3 cta_shape(CTA_ROWS * LANES_PER_STRIPE, 1, 1);
    dim3 grid_shape(m_extent / CTA_ROWS, 1, batch_count);
    BlockscaledWarpKernel<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS><<<grid_shape, cta_shape, 0, 0>>>(kernel_params);
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS>
inline void launch_blockscaled_cooperative(int m_extent, int batch_count, const BlockscaledParams& kernel_params)
{
    dim3 cta_shape(CTA_ROWS * LANES_PER_STRIPE, 1, 1);
    dim3 grid_shape(m_extent / CTA_ROWS, 1, batch_count);
    BlockscaledCooperativeKernel<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS><<<grid_shape, cta_shape, 0, 0>>>(kernel_params);
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS, bool USE_COOPERATIVE>
inline void select_and_launch_tile(int m_extent, int batch_count, BlockscaledParams& kernel_params)
{
    if constexpr (STATIC_TILE_ITERS == 0) {
        kernel_params.runtime_tile_iters = kernel_params.k_extent / (LANES_PER_STRIPE * 16);
    } else {
        kernel_params.runtime_tile_iters = 0;
    }
    if constexpr (USE_COOPERATIVE) {
        launch_blockscaled_cooperative<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS>(m_extent, batch_count, kernel_params);
    } else {
        launch_blockscaled_warp<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS>(m_extent, batch_count, kernel_params);
    }
}

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS, bool USE_COOPERATIVE>
inline void launch_tile(int m_extent, int batch_count, BlockscaledParams& kernel_params)
{
    select_and_launch_tile<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS, USE_COOPERATIVE>(
        m_extent, batch_count, kernel_params);
}

using TileLauncherFn = void (*)(int, int, BlockscaledParams&);

struct TileDispatchEntry {
    int k_extent;
    int required_m_multiple;  // 0 -> no requirement
    TileLauncherFn launch;
};

template <int CTA_ROWS, int LANES_PER_STRIPE, int STATIC_TILE_ITERS, bool USE_COOPERATIVE>
constexpr TileDispatchEntry make_tile_case(int k_extent, int required_m_multiple = CTA_ROWS)
{
    return TileDispatchEntry{
        k_extent,
        required_m_multiple,
        launch_tile<CTA_ROWS, LANES_PER_STRIPE, STATIC_TILE_ITERS, USE_COOPERATIVE>};
}

constexpr TileLauncherFn kGenericTileLauncher =
    launch_tile<8, 16, 0, false>;

constexpr TileDispatchEntry kSpecialTileCases[] = {
    make_tile_case<4, 32, 7, false>(3584, 0),
    make_tile_case<1, 128, 8192 / (128 * 16), true>(8192, 0),
    make_tile_case<1, 32, 4, false>(2048),
    make_tile_case<64, 16, 28, false>(7168),
    make_tile_case<2, 16, 64, false>(16384),
    make_tile_case<8, 16, 4, false>(1024, 0),
};


torch::Tensor gemv(torch::Tensor A, torch::Tensor B, torch::Tensor SFA, torch::Tensor SFB, torch::Tensor C)
{

    const auto problem_size = A.sizes();
    const int m_extent = problem_size[0];
    const int k_extent = problem_size[1];
    const int batch_count = problem_size[2];

    BlockscaledParams kernel_params{};
    kernel_params.k_extent = k_extent;
    kernel_params.runtime_tile_iters = 0;

    kernel_params.ptr_A = A.data_ptr();
    kernel_params.ptr_B = B.data_ptr();
    kernel_params.ptr_SFA = SFA.data_ptr();
    kernel_params.ptr_SFB = SFB.data_ptr();
    kernel_params.ptr_C = C.data_ptr();

    kernel_params.batch_stride_A = A.stride(2);
    kernel_params.batch_stride_B = B.stride(2);
    kernel_params.batch_stride_SFA = SFA.stride(2);
    kernel_params.batch_stride_SFB = SFB.stride(2);
    kernel_params.batch_stride_C = C.stride(2);

    kernel_params.stride_A = A.stride(0);
    kernel_params.stride_SFA = SFA.stride(0);

    if (kernel_params.k_extent <= 256) {
        launch_tile<16, 8, 0, false>(m_extent, batch_count, kernel_params);
        return C;
    }

    for (const auto& tile_case : kSpecialTileCases) {
        if (kernel_params.k_extent == tile_case.k_extent) {
            const bool m_match = (tile_case.required_m_multiple == 0) ||
                                 ((m_extent % tile_case.required_m_multiple) == 0);
            if (m_match) {
                tile_case.launch(m_extent, batch_count, kernel_params);
                return C;
            }
        }
    }

    kGenericTileLauncher(m_extent, batch_count, kernel_params);


    return C;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor gemv(torch::Tensor A, torch::Tensor B, torch::Tensor SFA, torch::Tensor SFB, torch::Tensor C);
"""

CUDA_FLAGS = [
    "-gencode=arch=compute_100a,code=sm_100a",
    "--ptxas-options=--gpu-name=sm_100a",
    "-O3",
    "-w",
    "-use_fast_math",
    "-maxrregcount=45",
]

LD_FLAGS = [
    "-lcublas",
    "-lcuda",
]

nvfp4_blockscaled = load_inline(
    name="nvfp4_gemv_blockscaled",
    cuda_sources=[CUDA_SRC],
    cpp_sources=[CPP_SRC],
    functions=["gemv"],
    extra_cuda_cflags=CUDA_FLAGS,
    extra_ldflags=LD_FLAGS,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, _, _, c = data
    return nvfp4_blockscaled.gemv(a, b, sfa, sfb, c)
