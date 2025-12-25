#!POPCORN leaderboard nvfp4_dual_gemm
#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

__device__
uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\\n\\t"
        ".reg .pred %%px;\\n\\t"
        "elect.sync _|%%px, %1;\\n\\t"
        "@%%px mov.s32 %0, 1;\\n\\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

__device__ inline
uint32_t get_cluster_ctarank() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ inline
void barrier_cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;");
}

__device__ inline
void barrier_cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;");
}

__device__ inline
void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__
void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\\n\\t"
        ".reg .pred P1;\\n\\t"
        "LAB_WAIT:\\n\\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\\n\\t"
        "@P1 bra.uni DONE;\\n\\t"
        "bra.uni LAB_WAIT;\\n\\t"
        "DONE:\\n\\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ inline
void mbarrier_expect_tx(int mbar_addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 _, [%0], %1;"
                            :: "r"(mbar_addr), "r"(size) : "memory");
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__device__ inline
void issue_tma(
    int smem, int stage_id, int iter_k,
    const CUtensorMap *A_tmap, const CUtensorMap *B_tmap,
    const char *SFA_ptr, const char *SFB_ptr,
    int off_m, int off_n, int K,
    int mbar_addr, uint64_t cache_A, uint64_t cache_B
) {
    constexpr int A_size = BLOCK_M * BLOCK_K / 2;
    constexpr int B_FRAG_N = BLOCK_N / 2;
    constexpr int B_size = B_FRAG_N * BLOCK_K / 2;
    constexpr int SFA_size = 128 * BLOCK_K / 16;
    constexpr int SFB_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

    // TMA tile height is capped at 128
    constexpr int TMA_M = (BLOCK_M > 128) ? 128 : BLOCK_M;
    constexpr int TMA_N = (B_FRAG_N > 128) ? 128 : B_FRAG_N;
    constexpr int A_TMA_SIZE = TMA_M * BLOCK_K / 2;
    constexpr int B_TMA_SIZE = TMA_N * BLOCK_K / 2;
    constexpr int NUM_A_TILES = BLOCK_M / TMA_M;
    constexpr int NUM_B_TILES = B_FRAG_N / TMA_N;

    const int A_smem = smem + stage_id * STAGE_SIZE;
    const int B_smem = A_smem + A_size;
    const int SFA_smem = B_smem + B_size;
    const int SFB_smem = SFA_smem + SFA_size;

    const int off_k = iter_k * BLOCK_K;
    const int rest_k = K / 16 / 4;

    mbarrier_expect_tx(mbar_addr, STAGE_SIZE);

    #pragma unroll
    for (int t = 0; t < NUM_A_TILES; t++) {
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
            "[%0], [%1, {%2, %3, %4}], [%5], %6;"
            :: "r"(A_smem + t * A_TMA_SIZE),
               "l"(A_tmap),
               "r"(0), "r"(off_m + t * TMA_M), "r"(off_k / 256),
               "r"(mbar_addr),
               "l"(cache_A)
            : "memory"
        );
    }
    const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(SFA_smem),
           "l"(SFA_src),
           "r"(SFA_size),
           "r"(mbar_addr),
           "l"(cache_A)
        : "memory"
    );

    #pragma unroll
    for (int t = 0; t < NUM_B_TILES; t++) {
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
            "[%0], [%1, {%2, %3, %4}], [%5], %6;"
            :: "r"(B_smem + t * B_TMA_SIZE),
               "l"(B_tmap),
               "r"(0), "r"(off_n + t * TMA_N), "r"(off_k / 256),
               "r"(mbar_addr),
               "l"(cache_B)
            : "memory"
        );
    }
    const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :: "r"(SFB_smem),
           "l"(SFB_src),
           "r"(SFB_size),
           "r"(mbar_addr),
           "l"(cache_B)
        : "memory"
    );
}

__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline
void tcgen05_mma_nvfp4(uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int scale_A_tmem, int scale_B_tmem, int enable_input_d) {
    const int d_tmem = 0;
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %6, 0;\\n\\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
             "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
    );
}

__device__ inline
void tcgen05_commit(int mbar_addr, uint16_t ctamask = 0x3) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.b64 [%0], %1;"
                            :: "r"(mbar_addr), "h"(ctamask) : "memory");
}

struct SHAPE {
    static constexpr char _32x32b[]  = ".32x32b";
    static constexpr char _16x128b[] = ".16x128b";
    static constexpr char _16x256b[] = ".16x256b";
};

struct NUM {
    static constexpr char x4[]  = ".x4";
    static constexpr char x8[]  = ".x8";
    static constexpr char x16[] = ".x16";
    static constexpr char x32[] = ".x32";
    static constexpr char x64[] = ".x64";
    static constexpr char x128[] = ".x128";
};

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_32regs(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%33%34.b32 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        "  %8,  %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
            "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
            "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
            "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
        : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_64regs(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%65%66.b32 "
                "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                " %16, %17, %18, %19, %20, %21, %22, %23, "
                " %24, %25, %26, %27, %28, %29, %30, %31, "
                " %32, %33, %34, %35, %36, %37, %38, %39, "
                " %40, %41, %42, %43, %44, %45, %46, %47, "
                " %48, %49, %50, %51, %52, %53, %54, %55, "
                " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                    "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                    "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                    "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                    "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                    "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                    "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                    "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
                : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

__device__ inline void tcgen05_ld_32x32bx32(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_32x32b, NUM::x32>(tmp, row, col); }
__device__ inline void tcgen05_ld_32x32bx64(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_32x32b, NUM::x64>(tmp, row, col); }

__device__ inline
float silu(float x) {
    return x / (1.0f + expf(-x));
}

void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *error_msg_ptr;
    if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
        error_msg_ptr = "unable to get error string";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void init_AB_tmap(
    CUtensorMap *tmap,
    const char *ptr,
    uint64_t global_height, uint64_t global_width,
    uint32_t shared_height, uint32_t shared_width
) {
    constexpr uint32_t rank = 3;
    // TMA boxDim[1] is limited to 128 max, so cap shared_height
    uint32_t tma_height = (shared_height > 128) ? 128 : shared_height;
    uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
    uint64_t globalStrides[rank-1] = {global_width / 2, 128};
    uint32_t boxDim[rank]          = {256, tma_height, shared_width / 256};
    uint32_t elementStrides[rank]  = {1, 1, 1};

    auto err = cuTensorMapEncodeTiled(
        tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,
        (void *)ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    check_cu(err);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ __cluster_dims__(2)
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_silu_kernel_2sm(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const char *SFA_ptr,
    const char *SFB1_ptr,
    const char *SFB2_ptr,
    half *C_ptr,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;

    // Get cluster info
    const uint32_t ctarank = get_cluster_ctarank();
    const bool is_cta0 = (ctarank == 0);

    // Block indexing (2-CTA cluster covers 2xBLOCK_M rows)
    const int cluster_id = blockIdx.x / 2;
    const int grid_n = N / BLOCK_N;
    const int cluster_m = cluster_id / grid_n;
    const int bid_n = cluster_id % grid_n;
    const int base_m = cluster_m * (2 * BLOCK_M);
    const int off_m = base_m + int(ctarank) * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;
    const int num_iters = K / BLOCK_K;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int A_size = BLOCK_M * BLOCK_K / 2;
    constexpr int B_FRAG_N = BLOCK_N / 2;
    constexpr int B_size = B_FRAG_N * BLOCK_K / 2;
    constexpr int SFA_size = 128 * BLOCK_K / 16;
    constexpr int SFB_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;
    constexpr int EPILOGUE_OFFSET = NUM_STAGES * STAGE_SIZE;

    // Epilogue shared memory (CTA-local): store SiLU(GEMM1) for final multiply after GEMM2
    half *silu_smem = reinterpret_cast<half *>(smem_ptr + EPILOGUE_OFFSET);

    // mbars for each phase:
    // - tma completion (per-stage)
    // - mma completion (per-stage)
    // - mainloop completion (final)
    // - CTA1 -> CTA0 peer-ready signals (per-stage)
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[NUM_STAGES * 6 + 2];
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int tma1_mbar_addr = mbar_base;
    const int mma1_mbar_addr = tma1_mbar_addr + NUM_STAGES * 8;
    const int main1_mbar_addr = mma1_mbar_addr + NUM_STAGES * 8;
    const int peer1_mbar_addr = main1_mbar_addr + 8;
    const int tma2_mbar_addr = peer1_mbar_addr + NUM_STAGES * 8;
    const int mma2_mbar_addr = tma2_mbar_addr + NUM_STAGES * 8;
    const int main2_mbar_addr = mma2_mbar_addr + NUM_STAGES * 8;
    const int peer2_mbar_addr = main2_mbar_addr + 8;

    constexpr int SFA_tmem = BLOCK_N;
    constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

    // Barrier cluster sync to ensure both CTAs are ready
    barrier_cluster_arrive();
    barrier_cluster_wait();

    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES * 6 + 2; i++)
            mbarrier_init(tma1_mbar_addr + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(BLOCK_N * 2));
    }
    __syncthreads();
    barrier_cluster_arrive();
    barrier_cluster_wait();

    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };
	    auto make_desc_SF = [](int addr) -> uint64_t {
	        const int SBO = 8 * 16;
	        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
	    };

    constexpr uint32_t i_desc =
        (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)(2 * BLOCK_M) >> 7U << 27U);

    constexpr int TILES_M_128 = 128 / BLOCK_M;
    constexpr int TILES_N_128 = 128 / BLOCK_N;
    const int tile_m_in_128 = (off_m / BLOCK_M) % TILES_M_128;
    const int tile_n_in_128 = (off_n / BLOCK_N) % TILES_N_128;

    // --- GEMM1 (A @ B1) ---
    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; iter_k++) {
            issue_tma<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, iter_k, iter_k, &A_tmap, &B1_tmap, SFA_ptr, SFB1_ptr,
                off_m, off_n + int(ctarank) * (BLOCK_N / 2), K, tma1_mbar_addr + iter_k * 8, cache_A, cache_B);
        }
        for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            mbarrier_wait(mma1_mbar_addr + stage_id * 8, (iter_k / NUM_STAGES - 1) % 2);
            issue_tma<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, stage_id, iter_k, &A_tmap, &B1_tmap, SFA_ptr, SFB1_ptr,
                off_m, off_n + int(ctarank) * (BLOCK_N / 2), K, tma1_mbar_addr + stage_id * 8, cache_A, cache_B);
        }
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync() && is_cta0) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int phase = (iter_k / NUM_STAGES) % 2;

            mbarrier_wait(tma1_mbar_addr + stage_id * 8, phase);
            mbarrier_wait(peer1_mbar_addr + stage_id * 8, phase);

            const int A_smem = smem + stage_id * STAGE_SIZE;
            const int B_smem = A_smem + A_size;
            const int SFA_smem = B_smem + B_size;
            const int SFB_smem = SFA_smem + SFA_size;

            constexpr uint64_t SF_desc = make_desc_SF(0);
            const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

            for (int k = 0; k < BLOCK_K / MMA_K; k++) {
                tcgen05_cp_nvfp4(SFA_tmem + k * 4, SFA_desc + (uint64_t)k * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB_tmem + k * 4, SFB_desc + (uint64_t)k * (512ULL >> 4ULL));
            }

            for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
                for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                    uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                    uint64_t b_desc = make_desc_AB(B_smem + k1 * B_FRAG_N * 128 + k2 * 32);
                    int k_sf = k1 * 4 + k2;
                    const int scale_A_tmem = SFA_tmem + k_sf * 4 + tile_m_in_128 * (BLOCK_M / 32);
                    const int scale_B_tmem = SFB_tmem + k_sf * 4 + tile_n_in_128 * (BLOCK_N / 32);
                    const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
                    tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
                }

            tcgen05_commit(mma1_mbar_addr + stage_id * 8);
        }
        tcgen05_commit(main1_mbar_addr);
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync() && !is_cta0) {
        uint32_t peer1_remote_base;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer1_remote_base) : "r"(static_cast<uint32_t>(peer1_mbar_addr)));

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int phase = (iter_k / NUM_STAGES) % 2;
            mbarrier_wait(tma1_mbar_addr + stage_id * 8, phase);
            asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                         :: "r"(peer1_remote_base + stage_id * 8) : "memory");
        }

        mbarrier_wait(main1_mbar_addr, 0);
    }

    __syncthreads();
    mbarrier_wait(main1_mbar_addr, 0);

    if (tid < BLOCK_M) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        constexpr int WIDTH = 64;
        for (int n = 0; n < BLOCK_N / WIDTH; n++) {
            float tmp[WIDTH];
            tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            for (int i = 0; i < WIDTH; i++) {
                silu_smem[tid * BLOCK_N + n * WIDTH + i] = __float2half(silu(tmp[i]));
            }
        }
    }

    __syncthreads();
    barrier_cluster_arrive();
    barrier_cluster_wait();

    // --- GEMM2 (A @ B2) ---
    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; iter_k++) {
            issue_tma<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, iter_k, iter_k, &A_tmap, &B2_tmap, SFA_ptr, SFB2_ptr,
                off_m, off_n + int(ctarank) * (BLOCK_N / 2), K, tma2_mbar_addr + iter_k * 8, cache_A, cache_B);
        }
        for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            mbarrier_wait(mma2_mbar_addr + stage_id * 8, (iter_k / NUM_STAGES - 1) % 2);
            issue_tma<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, stage_id, iter_k, &A_tmap, &B2_tmap, SFA_ptr, SFB2_ptr,
                off_m, off_n + int(ctarank) * (BLOCK_N / 2), K, tma2_mbar_addr + stage_id * 8, cache_A, cache_B);
        }
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync() && is_cta0) {
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int phase = (iter_k / NUM_STAGES) % 2;

            mbarrier_wait(tma2_mbar_addr + stage_id * 8, phase);
            mbarrier_wait(peer2_mbar_addr + stage_id * 8, phase);

            const int A_smem = smem + stage_id * STAGE_SIZE;
            const int B_smem = A_smem + A_size;
            const int SFA_smem = B_smem + B_size;
            const int SFB_smem = SFA_smem + SFA_size;

            constexpr uint64_t SF_desc = make_desc_SF(0);
            const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

            for (int k = 0; k < BLOCK_K / MMA_K; k++) {
                tcgen05_cp_nvfp4(SFA_tmem + k * 4, SFA_desc + (uint64_t)k * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB_tmem + k * 4, SFB_desc + (uint64_t)k * (512ULL >> 4ULL));
            }

            for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
                for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                    uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                    uint64_t b_desc = make_desc_AB(B_smem + k1 * B_FRAG_N * 128 + k2 * 32);
                    int k_sf = k1 * 4 + k2;
                    const int scale_A_tmem = SFA_tmem + k_sf * 4 + tile_m_in_128 * (BLOCK_M / 32);
                    const int scale_B_tmem = SFB_tmem + k_sf * 4 + tile_n_in_128 * (BLOCK_N / 32);
                    const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
                    tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
                }

            tcgen05_commit(mma2_mbar_addr + stage_id * 8);
        }
        tcgen05_commit(main2_mbar_addr);
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync() && !is_cta0) {
        uint32_t peer2_remote_base;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer2_remote_base) : "r"(static_cast<uint32_t>(peer2_mbar_addr)));

        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int phase = (iter_k / NUM_STAGES) % 2;
            mbarrier_wait(tma2_mbar_addr + stage_id * 8, phase);
            asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                         :: "r"(peer2_remote_base + stage_id * 8) : "memory");
        }

        mbarrier_wait(main2_mbar_addr, 0);
    }

    __syncthreads();
    mbarrier_wait(main2_mbar_addr, 0);

    if (tid < BLOCK_M) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        constexpr int WIDTH = 64;
        for (int n = 0; n < BLOCK_N / WIDTH; n++) {
            float tmp[WIDTH];
            tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            for (int i = 0; i < WIDTH; i++) {
                const int n_idx = n * WIDTH + i;
                float result = __half2float(silu_smem[tid * BLOCK_N + n_idx]) * tmp[i];
                C_ptr[(off_m + tid) * N + off_n + n_idx] = __float2half(result);
            }
        }
    }

    __syncthreads();
    barrier_cluster_arrive();
    barrier_cluster_wait();

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));

    barrier_cluster_arrive();
    barrier_cluster_wait();
}

at::Tensor dual_gemm_silu(
    const at::Tensor& A,
    const at::Tensor& B1,
    const at::Tensor& B2,
    const at::Tensor& SFA,
    const at::Tensor& SFB1,
    const at::Tensor& SFB2,
                at::Tensor& C
) {
    const int M = A.size(0);
    const int N = B1.size(0);
    const int K = A.size(1) * 2;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 6;

    auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr  = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr  = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());

    CUtensorMap A_tmap, B1_tmap, B2_tmap;
    init_AB_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
    init_AB_tmap(&B1_tmap, B1_ptr, N, K, BLOCK_N / 2, BLOCK_K);
    init_AB_tmap(&B2_tmap, B2_ptr, N, K, BLOCK_N / 2, BLOCK_K);

    // Each 2-CTA cluster covers (2*BLOCK_M) x BLOCK_N
    int num_tiles = (M / (2 * BLOCK_M)) * (N / BLOCK_N);
    int tb_size = BLOCK_M + 2 * WARP_SIZE;

    int AB_size = (BLOCK_M + (BLOCK_N / 2)) * (BLOCK_K / 2);
    int SFAB_size = 128 * (BLOCK_K / 16) * 2;
    int pipeline_size = (AB_size + SFAB_size) * NUM_STAGES;
    int epilogue_size = BLOCK_M * BLOCK_N * sizeof(half);
    int smem_size = pipeline_size + epilogue_size;

    auto kernel = dual_gemm_silu_kernel_2sm<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;

    // Set up kernel attributes for cluster launch
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    if (smem_size > 48'000)
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Configure cluster launch
    // gridDim = num_tiles * 2 CTAs total, grouped into num_tiles clusters of 2
    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(num_tiles * 2, 1, 1);
    config.blockDim = dim3(tb_size, 1, 1);
    config.dynamicSmemBytes = smem_size;

    // Set cluster dimensions (2 CTAs per cluster)
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    // Launch with cluster configuration
    cudaLaunchKernelEx(&config, kernel,
        A_tmap, B1_tmap, B2_tmap,
        SFA_ptr, SFB1_ptr, SFB2_ptr,
        C_ptr, M, N, K);

    return C;
}

TORCH_LIBRARY(dual_gemm_2sm, m) {
    m.def("dual_gemm_silu(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("dual_gemm_silu", &dual_gemm_silu);
}
"""

load_inline(
        "dual_gemm_2sm",
        cpp_sources="",
        cuda_sources=cuda_src,
        verbose=True,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--relocatable-device-code=false",
                "-lineinfo",
                "-Xptxas=-v",
        ],
        extra_ldflags=["-lcuda"],
)

dual_gemm_silu = torch.ops.dual_gemm_2sm.dual_gemm_silu


def custom_kernel(data: input_t) -> output_t:
        a, b1, b2 = data[0], data[1], data[2]
        sfa_perm, sfb1_perm, sfb2_perm = data[6], data[7], data[8]
        c = data[9]

        return dual_gemm_silu(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c)
