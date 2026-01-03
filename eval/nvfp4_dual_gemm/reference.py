import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from task import input_t, output_t
from common.utils import make_match_reference
from common.scale_helpers import SF_VEC_SIZE, ceil_div, to_blocked, create_scale_factor_tensors_fp32


def ref_kernel(data: input_t) -> output_t:
    """PyTorch reference implementation of NVFP4 block-scaled dual GEMM with silu activation."""
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, _, _, _, c_ref = data
    m, n, l = c_ref.shape

    ref1 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)
    ref2 = torch.empty((l, m, n), dtype=torch.float32, device="cuda").permute(1, 2, 0)

    for l_idx in range(l):
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])

        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref1[:, :, l_idx] = res1

        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = res2

    c_ref = (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)
    return c_ref


def generate_input(m: int, n: int, k: int, l: int, seed: int):
    """Generate input tensors for NVFP4 block-scaled dual GEMM with silu activation."""
    torch.manual_seed(seed)

    def create_fp4_tensors(l, mn, k):
        ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")
        ref_i8 = ref_i8 & 0b1011_1011
        return ref_i8.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)

    a_ref = create_fp4_tensors(l, m, k)
    b1_ref = create_fp4_tensors(l, n, k)
    b2_ref = create_fp4_tensors(l, n, k)

    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    sf_k = ceil_div(k, SF_VEC_SIZE)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors_fp32(l, m, sf_k)
    sfb1_ref_cpu, sfb1_ref_permuted = create_scale_factor_tensors_fp32(l, n, sf_k)
    sfb2_ref_cpu, sfb2_ref_permuted = create_scale_factor_tensors_fp32(l, n, sf_k)

    return (
        a_ref, b1_ref, b2_ref,
        sfa_ref_cpu.to("cuda"), sfb1_ref_cpu.to("cuda"), sfb2_ref_cpu.to("cuda"),
        sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted,
        c_ref
    )


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
