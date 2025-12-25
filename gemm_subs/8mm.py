# NVFP4 Block-Scaled GEMM - Persistent kernel with warp specialization
# Adapted from NVIDIA CUTLASS example


from typing import Union


import torch
from task import input_t, output_t


import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr




# Kernel configuration parameters
# Tile sizes for M, N dimensions (K computed dynamically)
mma_tiler_mn = (128, 64)
# FP4 data type for A and B
ab_dtype = cutlass.Float4E2M1FN
# FP8 data type for scale factors
sf_dtype = cutlass.Float8E4M3FN
# FP16 output type
c_dtype = cutlass.Float16
# Scale factor block size (16 elements share one scale)
sf_vec_size = 16
# Cluster shape
cluster_shape_mn = (1, 1)
# Use 2-CTA instructions (for mma_tiler_mn[0] == 256)
use_2cta_instrs = mma_tiler_mn[0] == 256
cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE


# Warp specialization IDs
epilog_warp_id = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5
threads_per_cta = 32 * len((mma_warp_id, tma_warp_id, *epilog_warp_id))


# Barriers
epilog_sync_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=32 * len(epilog_warp_id),
)
tmem_alloc_barrier = pipeline.NamedBarrier(
    barrier_id=2,
    num_threads=32 * len((mma_warp_id, *epilog_warp_id)),
)


# Memory configuration
smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
num_tmem_alloc_cols = 512
occupancy = 1
acc_dtype = cutlass.Float32


def ceil_div(a, b):
    return (a + b - 1) // b



@cute.kernel
def kernel_no_loop(
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
    epi_tile: cute.Tile,
    mma_tiler: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    cta_tile_shape_mnk: cutlass.Constexpr,
    num_ab_stage: cutlass.Constexpr[int],
    num_acc_stage: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    num_tma_load_bytes: cutlass.Constexpr[int],
    num_mcast_ctas_a: cutlass.Constexpr[int],
    num_mcast_ctas_b: cutlass.Constexpr[int],
    is_a_mcast: cutlass.Constexpr[bool],
    is_b_mcast: cutlass.Constexpr[bool],
    overlapping_accum: cutlass.Constexpr[bool],
    num_accumulator_tmem_cols: cutlass.Constexpr[int],
    num_sfa_tmem_cols: cutlass.Constexpr[int],
    num_sf_tmem_cols: cutlass.Constexpr[int],
    epi_tile_n: cutlass.Constexpr[int],
    iter_acc_early_release_in_epilogue: cutlass.Constexpr[int],
    c_layout: cutlass.Constexpr,
    shared_storage: cutlass.Constexpr,
):
    """GPU device kernel with warp specialization and persistent scheduling."""
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)


    # Prefetch TMA descriptors
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)
        cpasync.prefetch_descriptor(tma_atom_c)


    use_2cta = cute.size(tiled_mma.thr_id.shape) == 2


    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
    tidx, _, _ = cute.arch.thread_idx()


    # Allocate shared storage
    smem = utils.SmemAllocator()
    storage = smem.allocate(shared_storage)


    # Initialize pipelines
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
   )


    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_acc_consumer_threads = len(epilog_warp_id) * (2 if use_2cta else 1)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
   )


    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilog_warp_id[0],
        is_two_cta=use_2cta,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
   )


    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)


    # Setup SMEM tensors
    sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)


    # Multicast masks
    a_full_mcast_mask = None
    b_full_mcast_mask = None
    sfa_full_mcast_mask = None
    sfb_full_mcast_mask = None
    if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
        )


    # Partition global tensors
    gA_mkl = cute.local_tile(mA_mkl, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gB_nkl = cute.local_tile(mB_nkl, cute.slice_(mma_tiler, (0, None, None)), (None, None, None))
    gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gSFB_nkl = cute.local_tile(mSFB_nkl, cute.slice_(mma_tiler_sfb, (0, None, None)), (None, None, None))
    gC_mnl = cute.local_tile(mC_mnl, cute.slice_(mma_tiler, (None, None, 0)), (None, None, None))
    k_tile_cnt = cute.size(gA_mkl, mode=[3])


    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
    tCgC = thr_mma.partition_C(gC_mnl)


    # TMA partitions
    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
   )
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
   )
    sfa_cta_layout = a_cta_layout
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
   )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
   )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)


    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    if cutlass.const_expr(overlapping_accum):
        num_acc_stage_overlapped = 2
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (256 - num_sf_tmem_cols) * tCtAcc_fake.stride[0][1]
                )
            )
        )
    else:
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))


    pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)


    # TMA warp
    if warp_idx == tma_warp_id:
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage
        )

        tAgA_slice = tAgA[(None, 0, None, 0)]
        tBgB_slice = tBgB[(None, bidx, None, 0)]
        tAgSFA_slice = tAgSFA[(None, 0, None, 0)]
        slice_n = bidx
        if cutlass.const_expr(cta_tile_shape_mnk[1] == 64):
            slice_n = bidx // 2
        tBgSFB_slice = tBgSFB[(None, slice_n, None, 0)]

        ab_producer_state.reset_count()
        peek_ab_empty_status = cutlass.Boolean(1)
        if ab_producer_state.count < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)


        for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            cute.copy(tma_atom_a, tAgA_slice[(None, ab_producer_state.count)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=a_full_mcast_mask)
            cute.copy(tma_atom_b, tBgB_slice[(None, ab_producer_state.count)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=b_full_mcast_mask)
            cute.copy(tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)],
                    tAsSFA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfa_full_mcast_mask)
            cute.copy(tma_atom_sfb, tBgSFB_slice[(None, ab_producer_state.count)],
                    tBsSFB[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    mcast_mask=sfb_full_mcast_mask)
            ab_producer_state.advance()
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

        ab_pipeline.producer_tail(ab_producer_state)


    # MMA warp
    if warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)


        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + num_accumulator_tmem_cols, dtype=sf_dtype)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler, sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)


        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols, dtype=sf_dtype
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler, sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)


        # S2T copy for SFA
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        copy_atom_s2t_sfa = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t_sfa, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_)
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)


        # S2T copy for SFB
        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        copy_atom_s2t_sfb = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t_sfb, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_)
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_ab_stage)
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_acc_stage)

        if cutlass.const_expr(overlapping_accum):
            acc_stage_index = acc_producer_state.phase ^ 1
        else:
            acc_stage_index = acc_producer_state.index
        tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]


        ab_consumer_state.reset_count()
        peek_ab_full_status = cutlass.Boolean(1)
        if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)


        if is_leader_cta:
            acc_pipeline.producer_acquire(acc_producer_state)


        tCtSFB_mma = tCtSFB
        if cutlass.const_expr(cta_tile_shape_mnk[1] == 192):
            offset = cutlass.Int32(2) if bidx % 2 == 1 else cutlass.Int32(0)
            shifted_ptr = cute.recast_ptr(
                acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols + offset,
                dtype=sf_dtype,
            )
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
        elif cutlass.const_expr(cta_tile_shape_mnk[1] == 64):
            offset = cutlass.Int32((bidx % 2) * 2)
            shifted_ptr = cute.recast_ptr(
                acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols + offset,
                dtype=sf_dtype,
            )
            tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)


        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)


        for k_tile in range(k_tile_cnt):
            if is_leader_cta:
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)


                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                    sf_kblock_coord = (None, None, kblock_idx)
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)


                ab_pipeline.consumer_release(ab_consumer_state)


            ab_consumer_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)


        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)
        acc_producer_state.advance()


        acc_pipeline.producer_tail(acc_producer_state)


    # Epilogue warps
    if warp_idx < mma_warp_id:
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)


        # Epilogue TMEM copy setup
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk, c_layout, c_dtype, acc_dtype, epi_tile, use_2cta,
        )
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, c_dtype)


        # R2S copy setup
        copy_atom_r2s = sm100_utils.get_smem_store_op(c_layout, c_dtype, acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)


        # GMEM copy setup
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma)


        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_acc_stage)

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(epilog_warp_id))
        c_pipeline = pipeline.PipelineTmaStore.create(num_stages=num_c_stage, producer_group=c_producer_group)

        bSG_gC = bSG_gC_partitioned[(None, None, None, 0, bidx, 0)]


        if cutlass.const_expr(overlapping_accum):
            acc_stage_index = acc_consumer_state.phase
            reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
        else:
            acc_stage_index = acc_consumer_state.index


        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
        acc_pipeline.consumer_wait(acc_consumer_state)


        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))


        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        num_prev_subtiles = 0
        for subtile_idx in cutlass.range(subtile_cnt):
            real_subtile_idx = subtile_idx
            if cutlass.const_expr(overlapping_accum):
                if reverse_subtile:
                    real_subtile_idx = cta_tile_shape_mnk[1] // epi_tile_n - 1 - subtile_idx


            tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)


            if cutlass.const_expr(overlapping_accum):
                if subtile_idx == iter_acc_early_release_in_epilogue:
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()


            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            acc_vec = acc_vec.to(c_dtype)
            tRS_rC.store(acc_vec)


            c_buffer = (num_prev_subtiles + real_subtile_idx) % num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
            epilog_sync_barrier.arrive_and_wait()


            if warp_idx == epilog_warp_id[0]:
                cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()


        if cutlass.const_expr(not overlapping_accum):
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer_state)
            acc_consumer_state.advance()

        tmem.relinquish_alloc_permit()
        epilog_sync_barrier.arrive_and_wait()
        tmem.free(acc_tmem_ptr)
        c_pipeline.producer_tail()

    return


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tiled_mma_sfb: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_sfa: cute.CopyAtom,
    mSFA_mkl: cute.Tensor,
    tma_atom_sfb: cute.CopyAtom,
    mSFB_nkl: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
    epi_tile: cute.Tile,
    tile_sched_params: utils.PersistentTileSchedulerParams,
    mma_tiler: cutlass.Constexpr,
    mma_tiler_sfb: cutlass.Constexpr,
    cta_tile_shape_mnk: cutlass.Constexpr,
    num_ab_stage: cutlass.Constexpr[int],
    num_acc_stage: cutlass.Constexpr[int],
    num_c_stage: cutlass.Constexpr[int],
    num_tma_load_bytes: cutlass.Constexpr[int],
    num_mcast_ctas_a: cutlass.Constexpr[int],
    num_mcast_ctas_b: cutlass.Constexpr[int],
    is_a_mcast: cutlass.Constexpr[bool],
    is_b_mcast: cutlass.Constexpr[bool],
    overlapping_accum: cutlass.Constexpr[bool],
    num_accumulator_tmem_cols: cutlass.Constexpr[int],
    num_sfa_tmem_cols: cutlass.Constexpr[int],
    num_sf_tmem_cols: cutlass.Constexpr[int],
    epi_tile_n: cutlass.Constexpr[int],
    iter_acc_early_release_in_epilogue: cutlass.Constexpr[int],
    c_layout: cutlass.Constexpr,
    shared_storage: cutlass.Constexpr,
):
    """GPU device kernel with warp specialization and persistent scheduling."""
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)


    # Prefetch TMA descriptors
    if warp_idx == tma_warp_id:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_sfa)
        cpasync.prefetch_descriptor(tma_atom_sfb)
        cpasync.prefetch_descriptor(tma_atom_c)


    use_2cta = cute.size(tiled_mma.thr_id.shape) == 2


    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = mma_tile_coord_v == 0
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
    tidx, _, _ = cute.arch.thread_idx()


    # Allocate shared storage
    smem = utils.SmemAllocator()
    storage = smem.allocate(shared_storage)


    # Initialize pipelines
    ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
    ab_pipeline = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_pipeline_producer_group,
        consumer_group=ab_pipeline_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
   )


    acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_acc_consumer_threads = len(epilog_warp_id) * (2 if use_2cta else 1)
    acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=acc_pipeline_producer_group,
        consumer_group=acc_pipeline_consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
   )


    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=epilog_warp_id[0],
        is_two_cta=use_2cta,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
   )


    pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)


    # Setup SMEM tensors
    sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)


    # Multicast masks
    a_full_mcast_mask = None
    b_full_mcast_mask = None
    sfa_full_mcast_mask = None
    sfb_full_mcast_mask = None
    if cutlass.const_expr(is_a_mcast or is_b_mcast or use_2cta):
        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
        )


    # Partition global tensors
    gA_mkl = cute.local_tile(mA_mkl, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gB_nkl = cute.local_tile(mB_nkl, cute.slice_(mma_tiler, (0, None, None)), (None, None, None))
    gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None))
    gSFB_nkl = cute.local_tile(mSFB_nkl, cute.slice_(mma_tiler_sfb, (0, None, None)), (None, None, None))
    gC_mnl = cute.local_tile(mC_mnl, cute.slice_(mma_tiler, (None, None, 0)), (None, None, None))
    k_tile_cnt = cute.size(gA_mkl, mode=[3])


    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgSFA = thr_mma.partition_A(gSFA_mkl)
    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
    tCgC = thr_mma.partition_C(gC_mnl)


    # TMA partitions
    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
   )
    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
   )
    sfa_cta_layout = a_cta_layout
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
        cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
   )
    tAsSFA = cute.filter_zeros(tAsSFA)
    tAgSFA = cute.filter_zeros(tAgSFA)
    sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
    tBsSFB, tBgSFB = cpasync.tma_partition(
        tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
        cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
   )
    tBsSFB = cute.filter_zeros(tBsSFB)
    tBgSFB = cute.filter_zeros(tBgSFB)


    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    if cutlass.const_expr(overlapping_accum):
        num_acc_stage_overlapped = 2
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage_overlapped))
        tCtAcc_fake = cute.make_tensor(
            tCtAcc_fake.iterator,
            cute.make_layout(
                tCtAcc_fake.shape,
                stride=(
                    tCtAcc_fake.stride[0],
                    tCtAcc_fake.stride[1],
                    tCtAcc_fake.stride[2],
                    (256 - num_sf_tmem_cols) * tCtAcc_fake.stride[0][1]
                )
            )
        )
    else:
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))


    pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)


    # TMA warp
    if warp_idx == tma_warp_id:
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_ab_stage
        )


        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
            tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
            slice_n = mma_tile_coord_mnl[1]
            if cutlass.const_expr(cta_tile_shape_mnk[1] == 64):
                slice_n = mma_tile_coord_mnl[1] // 2
            tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]


            ab_producer_state.reset_count()
            peek_ab_empty_status = cutlass.Boolean(1)
            if ab_producer_state.count < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)


            for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
                cute.copy(tma_atom_a, tAgA_slice[(None, ab_producer_state.count)],
                         tAsA[(None, ab_producer_state.index)],
                         tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                         mcast_mask=a_full_mcast_mask)
                cute.copy(tma_atom_b, tBgB_slice[(None, ab_producer_state.count)],
                         tBsB[(None, ab_producer_state.index)],
                         tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                         mcast_mask=b_full_mcast_mask)
                cute.copy(tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)],
                         tAsSFA[(None, ab_producer_state.index)],
                         tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                         mcast_mask=sfa_full_mcast_mask)
                cute.copy(tma_atom_sfb, tBgSFB_slice[(None, ab_producer_state.count)],
                         tBsSFB[(None, ab_producer_state.index)],
                         tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                         mcast_mask=sfb_full_mcast_mask)
                ab_producer_state.advance()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)


            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        ab_pipeline.producer_tail(ab_producer_state)


    # MMA warp
    if warp_idx == mma_warp_id:
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)


        sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + num_accumulator_tmem_cols, dtype=sf_dtype)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler, sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)


        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols, dtype=sf_dtype
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler, sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)


        # S2T copy for SFA
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        copy_atom_s2t_sfa = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t_sfa, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfa, tCsSFA_compact_s2t_)
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)


        # S2T copy for SFB
        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        copy_atom_s2t_sfb = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(cta_group), sf_dtype)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t_sfb, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t_sfb, tCsSFB_compact_s2t_)
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)


        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_ab_stage)
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_acc_stage)


        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            if cutlass.const_expr(overlapping_accum):
                acc_stage_index = acc_producer_state.phase ^ 1
            else:
                acc_stage_index = acc_producer_state.index
            tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]


            ab_consumer_state.reset_count()
            peek_ab_full_status = cutlass.Boolean(1)
            if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)


            if is_leader_cta:
                acc_pipeline.producer_acquire(acc_producer_state)


            tCtSFB_mma = tCtSFB
            if cutlass.const_expr(cta_tile_shape_mnk[1] == 192):
                offset = cutlass.Int32(2) if mma_tile_coord_mnl[1] % 2 == 1 else cutlass.Int32(0)
                shifted_ptr = cute.recast_ptr(
                    acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols + offset,
                    dtype=sf_dtype,
                )
                tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
            elif cutlass.const_expr(cta_tile_shape_mnk[1] == 64):
                offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                shifted_ptr = cute.recast_ptr(
                    acc_tmem_ptr + num_accumulator_tmem_cols + num_sfa_tmem_cols + offset,
                    dtype=sf_dtype,
                )
                tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)


            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)


            for k_tile in range(k_tile_cnt):
                if is_leader_cta:
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)


                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_coord], tCrB[kblock_coord], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)


                    ab_pipeline.consumer_release(ab_consumer_state)


                ab_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)


            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
            acc_producer_state.advance()


            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        acc_pipeline.producer_tail(acc_producer_state)


    # Epilogue warps
    if warp_idx < mma_warp_id:
        tmem.allocate(num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(acc_dtype)
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)


        # Epilogue TMEM copy setup
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk, c_layout, c_dtype, acc_dtype, epi_tile, use_2cta,
        )
        tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, c_dtype)


        # R2S copy setup
        copy_atom_r2s = sm100_utils.get_smem_store_op(c_layout, c_dtype, acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)


        # GMEM copy setup
        gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma)


        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_acc_stage)


        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(epilog_warp_id))
        c_pipeline = pipeline.PipelineTmaStore.create(num_stages=num_c_stage, producer_group=c_producer_group)


        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]


            if cutlass.const_expr(overlapping_accum):
                acc_stage_index = acc_consumer_state.phase
                reverse_subtile = cutlass.Boolean(True) if acc_stage_index == 0 else cutlass.Boolean(False)
            else:
                acc_stage_index = acc_consumer_state.index


            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
            acc_pipeline.consumer_wait(acc_consumer_state)


            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))


            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
            for subtile_idx in cutlass.range(subtile_cnt):
                real_subtile_idx = subtile_idx
                if cutlass.const_expr(overlapping_accum):
                    if reverse_subtile:
                        real_subtile_idx = cta_tile_shape_mnk[1] // epi_tile_n - 1 - subtile_idx


                tTR_tAcc_mn = tTR_tAcc[(None, None, None, real_subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)


                if cutlass.const_expr(overlapping_accum):
                    if subtile_idx == iter_acc_early_release_in_epilogue:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()


                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                acc_vec = acc_vec.to(c_dtype)
                tRS_rC.store(acc_vec)


                c_buffer = (num_prev_subtiles + real_subtile_idx) % num_c_stage
                cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
                cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)
                epilog_sync_barrier.arrive_and_wait()


                if warp_idx == epilog_warp_id[0]:
                    cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, real_subtile_idx)])
                    c_pipeline.producer_commit()
                    c_pipeline.producer_acquire()
                epilog_sync_barrier.arrive_and_wait()


            if cutlass.const_expr(not overlapping_accum):
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()


            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()


        tmem.relinquish_alloc_permit()
        epilog_sync_barrier.arrive_and_wait()
        tmem.free(acc_tmem_ptr)
        c_pipeline.producer_tail()


    return

@cute.jit
def my_kernel_no_loop(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
    max_active_clusters: cutlass.Constexpr,
):
    """Host-side JIT function to prepare tensors and launch GPU kernel."""
    m, n, k, l = problem_size


    # Create tensors from pointers
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
             (m, cute.assume(k, 32), l),
             stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
             (n, cute.assume(k, 32), l),
             stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n))
    )


    # Setup sfa/sfb tensor
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)


    # Compute MMA configuration
    mma_inst_shape_mn = mma_tiler_mn
    mma_inst_shape_mn_sfb = (
        mma_inst_shape_mn[0] // (2 if use_2cta_instrs else 1),
        cute.round_up(mma_inst_shape_mn[1], 128),
    )


    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype,
        utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode(),
        utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode(),
        sf_dtype,
        sf_vec_size,
        cta_group,
        mma_inst_shape_mn,
    )
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype,
        utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode(),
        utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode(),
        sf_dtype,
        sf_vec_size,
        tcgen05.CtaGroup.ONE,
        mma_inst_shape_mn_sfb,
    )


    mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    mma_inst_tile_k = 4
    mma_tiler = (mma_inst_shape_mn[0], mma_inst_shape_mn[1], mma_inst_shape_k * mma_inst_tile_k)
    mma_tiler_sfb = (mma_inst_shape_mn_sfb[0], mma_inst_shape_mn_sfb[1], mma_inst_shape_k * mma_inst_tile_k)


    cta_tile_shape_mnk = (
        mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler[1],
        mma_tiler[2],
    )
    cta_tile_shape_mnk_sfb = (
        mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_sfb[1],
        mma_tiler_sfb[2],
    )

    # The k is hardcoded to 1 because we don't do split-k
    # if do split-k, we need to change this.
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )
    cluster_layout_sfb_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma_sfb.thr_id.shape,),
    )


    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
    is_a_mcast = num_mcast_ctas_a > 1
    is_b_mcast = num_mcast_ctas_b > 1


    c_layout = utils.LayoutEnum.from_tensor(c_tensor)
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk, use_2cta_instrs, c_layout, c_dtype,
    )
    epi_tile_n = cute.size(epi_tile[1])


    # Compute stages
    num_acc_stage = 1 if mma_tiler[1] == 256 else 2
    num_c_stage = 2
    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, ab_dtype, 1)
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, ab_dtype, 1)
    sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, 1)
    sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec_size, 1)
    c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
    ab_bytes_per_stage = (
        cute.size_in_bytes(ab_dtype, a_smem_layout_stage_one) +
        cute.size_in_bytes(ab_dtype, b_smem_layout_staged_one) +
        cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one) +
        cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
    )
    mbar_helpers_bytes = 1024
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
    c_bytes = c_bytes_per_stage * num_c_stage
    num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
    num_c_stage += (
        smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)
    ) // (occupancy * c_bytes_per_stage)


    # Compute SMEM layouts
    # the sm100_utils.make_smem_layout_a return ComposedLayout, which includes swizzle information,
    # so in the kernel, read from sC, sA, sB, need to apply the swizzle.
    # And the  blockscaled_utils.make_smem_layout_sfa returns a plain Layout, no swizzle information,
    # so in the kernel, read from sSFA, sSFB, no need to apply swizzle.
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, ab_dtype, num_ab_stage)
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, ab_dtype, num_ab_stage)
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, num_ab_stage)
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec_size, num_ab_stage)
    c_smem_layout_staged = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, num_c_stage)


    overlapping_accum = num_acc_stage == 1
    sf_atom_mn = 32
    num_sfa_tmem_cols = (cta_tile_shape_mnk[0] // sf_atom_mn) * 4
    num_sfb_tmem_cols = (cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * 4
    num_sf_tmem_cols = num_sfa_tmem_cols + num_sfb_tmem_cols
    num_accumulator_tmem_cols = (
        cta_tile_shape_mnk[1] * num_acc_stage
        if not overlapping_accum
        else cta_tile_shape_mnk[1] * 2 - num_sf_tmem_cols
    )
    iter_acc_early_release_in_epilogue = num_sf_tmem_cols // epi_tile_n


    atom_thr_size = cute.size(tiled_mma.thr_id.shape)


    # Setup TMA for A
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a_tensor, a_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
    )


    # Setup TMA for B
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_tensor, b_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
    )


    # Setup TMA for SFA
    sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sfa_op, sfa_tensor, sfa_smem_layout, mma_tiler, tiled_mma,
        cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
    )


    # Setup TMA for SFB
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma.thr_id)
    sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb_tensor, sfb_smem_layout, mma_tiler_sfb, tiled_mma_sfb,
        cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
    )


    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size


    # Setup TMA store for C
    epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, epi_tile,
    )


    # Compute grid
    # Calculate number of tiles
    num_n_tiles = (n + mma_tiler[1] - 1) // mma_tiler[1]
    grid = (num_n_tiles, 1, 1)

    buffer_align_bytes = 1024


    # Define shared storage
    @cute.struct
    class SharedStorage:
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32
        sC: cute.struct.Align[
            cute.struct.MemRange[c_dtype, cute.cosize(c_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[ab_dtype, cute.cosize(a_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[ab_dtype, cute.cosize(b_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sSFA: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfa_smem_layout_staged)],
            buffer_align_bytes,
        ]
        sSFB: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfb_smem_layout_staged)],
            buffer_align_bytes,
        ]
    # Launch kernel
    kernel_no_loop(
        tiled_mma, tiled_mma_sfb,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa,
        tma_atom_sfb, tma_tensor_sfb,
        tma_atom_c, tma_tensor_c,
        cluster_layout_vmnk,
        cluster_layout_sfb_vmnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        c_smem_layout_staged,
        epi_tile,
        mma_tiler,
        mma_tiler_sfb,
        cta_tile_shape_mnk,
        num_ab_stage,
        num_acc_stage,
        num_c_stage,
        num_tma_load_bytes,
        num_mcast_ctas_a,
        num_mcast_ctas_b,
        is_a_mcast,
        is_b_mcast,
        overlapping_accum,
        num_accumulator_tmem_cols,
        num_sfa_tmem_cols,
        num_sf_tmem_cols,
        epi_tile_n,
        iter_acc_early_release_in_epilogue,
        c_layout,
        SharedStorage,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(*cluster_shape_mn, 1),
        min_blocks_per_mp=1,
    )
    return


@cute.jit
def my_kernel(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    problem_size: tuple,
    max_active_clusters: cutlass.Constexpr,
):
    """Host-side JIT function to prepare tensors and launch GPU kernel."""
    m, n, k, l = problem_size


    # Create tensors from pointers
    a_tensor = cute.make_tensor(
        a_ptr,
        cute.make_layout(
             (m, cute.assume(k, 32), l),
             stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
        ),
    )
    b_tensor = cute.make_tensor(
        b_ptr,
        cute.make_layout(
             (n, cute.assume(k, 32), l),
             stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
        ),
    )
    c_tensor = cute.make_tensor(
        c_ptr, cute.make_layout((cute.assume(m, 32), n, l), stride=(n, 1, m * n))
    )


    # Setup sfa/sfb tensor
    sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, sf_vec_size)
    sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
    sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, sf_vec_size)
    sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)


    # Compute MMA configuration
    mma_inst_shape_mn = mma_tiler_mn
    mma_inst_shape_mn_sfb = (
        mma_inst_shape_mn[0] // (2 if use_2cta_instrs else 1),
        cute.round_up(mma_inst_shape_mn[1], 128),
    )


    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype,
        utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode(),
        utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode(),
        sf_dtype,
        sf_vec_size,
        cta_group,
        mma_inst_shape_mn,
    )
    tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        ab_dtype,
        utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode(),
        utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode(),
        sf_dtype,
        sf_vec_size,
        tcgen05.CtaGroup.ONE,
        mma_inst_shape_mn_sfb,
    )


    mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    mma_inst_tile_k = 4
    mma_tiler = (mma_inst_shape_mn[0], mma_inst_shape_mn[1], mma_inst_shape_k * mma_inst_tile_k)
    mma_tiler_sfb = (mma_inst_shape_mn_sfb[0], mma_inst_shape_mn_sfb[1], mma_inst_shape_k * mma_inst_tile_k)


    cta_tile_shape_mnk = (
        mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler[1],
        mma_tiler[2],
    )
    cta_tile_shape_mnk_sfb = (
        mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_sfb[1],
        mma_tiler_sfb[2],
    )

    # The k is hardcoded to 1 because we don't do split-k
    # if do split-k, we need to change this.
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma.thr_id.shape,),
    )
    cluster_layout_sfb_vmnk = cute.tiled_divide(
        cute.make_layout((*cluster_shape_mn, 1)),
        (tiled_mma_sfb.thr_id.shape,),
    )


    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
    is_a_mcast = num_mcast_ctas_a > 1
    is_b_mcast = num_mcast_ctas_b > 1


    c_layout = utils.LayoutEnum.from_tensor(c_tensor)
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        cta_tile_shape_mnk, use_2cta_instrs, c_layout, c_dtype,
    )
    epi_tile_n = cute.size(epi_tile[1])


    # Compute stages
    num_acc_stage = 1 if mma_tiler[1] == 256 else 2
    num_c_stage = 2
    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, ab_dtype, 1)
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, ab_dtype, 1)
    sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, 1)
    sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec_size, 1)
    c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)
    ab_bytes_per_stage = (
        cute.size_in_bytes(ab_dtype, a_smem_layout_stage_one) +
        cute.size_in_bytes(ab_dtype, b_smem_layout_staged_one) +
        cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one) +
        cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
    )
    mbar_helpers_bytes = 1024
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
    c_bytes = c_bytes_per_stage * num_c_stage
    num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
    num_c_stage += (
        smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + c_bytes)
    ) // (occupancy * c_bytes_per_stage)


    # Compute SMEM layouts
    # the sm100_utils.make_smem_layout_a return ComposedLayout, which includes swizzle information,
    # so in the kernel, read from sC, sA, sB, need to apply the swizzle.
    # And the  blockscaled_utils.make_smem_layout_sfa returns a plain Layout, no swizzle information,
    # so in the kernel, read from sSFA, sSFB, no need to apply swizzle.
    a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, ab_dtype, num_ab_stage)
    b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, ab_dtype, num_ab_stage)
    sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec_size, num_ab_stage)
    sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec_size, num_ab_stage)
    c_smem_layout_staged = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, num_c_stage)


    overlapping_accum = num_acc_stage == 1
    sf_atom_mn = 32
    num_sfa_tmem_cols = (cta_tile_shape_mnk[0] // sf_atom_mn) * 4
    num_sfb_tmem_cols = (cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * 4
    num_sf_tmem_cols = num_sfa_tmem_cols + num_sfb_tmem_cols
    num_accumulator_tmem_cols = (
        cta_tile_shape_mnk[1] * num_acc_stage
        if not overlapping_accum
        else cta_tile_shape_mnk[1] * 2 - num_sf_tmem_cols
    )
    iter_acc_early_release_in_epilogue = num_sf_tmem_cols // epi_tile_n


    atom_thr_size = cute.size(tiled_mma.thr_id.shape)


    # Setup TMA for A
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, a_tensor, a_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
    )


    # Setup TMA for B
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, b_tensor, b_smem_layout, mma_tiler, tiled_mma, cluster_layout_vmnk.shape,
    )


    # Setup TMA for SFA
    sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sfa_op, sfa_tensor, sfa_smem_layout, mma_tiler, tiled_mma,
        cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
    )


    # Setup TMA for SFB
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(cluster_shape_mn, tiled_mma.thr_id)
    sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))
    tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op, sfb_tensor, sfb_smem_layout, mma_tiler_sfb, tiled_mma_sfb,
        cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
    )


    # Compute TMA load bytes
    a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
    b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
    sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
    sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
    num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size


    # Setup TMA store for C
    epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, epi_tile,
    )


    # Compute grid
    c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
    gc = cute.zipped_divide(c_tensor, tiler=c_shape)
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    cluster_shape_mnl = (*cluster_shape_mn, 1)
    tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)


    buffer_align_bytes = 1024


    # Define shared storage
    @cute.struct
    class SharedStorage:
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage]
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32
        sC: cute.struct.Align[
            cute.struct.MemRange[c_dtype, cute.cosize(c_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sA: cute.struct.Align[
            cute.struct.MemRange[ab_dtype, cute.cosize(a_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[ab_dtype, cute.cosize(b_smem_layout_staged.outer)],
            buffer_align_bytes,
        ]
        sSFA: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfa_smem_layout_staged)],
            buffer_align_bytes,
        ]
        sSFB: cute.struct.Align[
            cute.struct.MemRange[sf_dtype, cute.cosize(sfb_smem_layout_staged)],
            buffer_align_bytes,
        ]
    # Launch kernel
    kernel(
        tiled_mma, tiled_mma_sfb,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_sfa, tma_tensor_sfa,
        tma_atom_sfb, tma_tensor_sfb,
        tma_atom_c, tma_tensor_c,
        cluster_layout_vmnk,
        cluster_layout_sfb_vmnk,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        c_smem_layout_staged,
        epi_tile,
        tile_sched_params,
        mma_tiler,
        mma_tiler_sfb,
        cta_tile_shape_mnk,
        num_ab_stage,
        num_acc_stage,
        num_c_stage,
        num_tma_load_bytes,
        num_mcast_ctas_a,
        num_mcast_ctas_b,
        is_a_mcast,
        is_b_mcast,
        overlapping_accum,
        num_accumulator_tmem_cols,
        num_sfa_tmem_cols,
        num_sf_tmem_cols,
        epi_tile_n,
        iter_acc_early_release_in_epilogue,
        c_layout,
        SharedStorage,
    ).launch(
        grid=grid,
        block=[threads_per_cta, 1, 1],
        cluster=(*cluster_shape_mn, 1),
        min_blocks_per_mp=1,
    )
    return




# Global cache for compiled kernel
_compiled_kernel_cache_loop = None
_compiled_kernel_cache_no_loop = None

def compile_kernel(use_loop: bool):
    """Compile the kernel once and cache it."""
    global _compiled_kernel_cache_loop, _compiled_kernel_cache_no_loop
    
    if use_loop:
        if _compiled_kernel_cache_loop is not None:
            return _compiled_kernel_cache_loop
    else:
        if _compiled_kernel_cache_no_loop is not None:
            return _compiled_kernel_cache_no_loop

    # Compute max active clusters
    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    # Create CuTe pointers
    a_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

    # Compile with the specific use_loop value (compile-time constant)
    if use_loop:
        compiled = cute.compile(
            my_kernel, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0),
            max_active_clusters, 
            options="--opt-level 2"
        )
    else:
        compiled = cute.compile(
            my_kernel_no_loop, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0),
            max_active_clusters, 
            options="--opt-level 2"
        )
    
    if use_loop:
         _compiled_kernel_cache_loop = compiled
    else:
         _compiled_kernel_cache_no_loop = compiled
    
    return compiled




def custom_kernel(data: input_t) -> output_t:
    """Execute the block-scaled GEMM kernel."""
    a, b, _, _, sfa_permuted, sfb_permuted, c = data
    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    n, _, _ = b.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2

    # Only use no_loop when m=128 AND tiles < 148
    use_loop = (m != 128) or ((m / mma_tiler_mn[0]) * (n / mma_tiler_mn[1]) > 148)
    compiled_func = compile_kernel(use_loop)

    # Create CuTe pointers
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ptr = make_ptr(c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)


    # Execute the compiled kernel
    compiled_func(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))
    return c
