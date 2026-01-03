import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from task import input_t, output_t
from common.utils import make_match_reference
from common.scale_helpers import SF_VEC_SIZE, ceil_div, to_blocked, create_scale_factor_tensors


def ref_kernel(data: input_t) -> output_t:
    """PyTorch reference implementation of NVFP4 block-scaled GEMV."""
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    _, _, l = c_ref.shape

    for l_idx in range(l):
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, 0, l_idx] = res[:, 0]
    return c_ref


def generate_input(m: int, k: int, l: int, seed: int):
    """Generate input tensors for NVFP4 block-scaled GEMV."""
    torch.manual_seed(seed)

    n = 1
    n_padded_128 = 128

    a_ref = torch.randint(0, 4, (l, m, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    b_ref = torch.randint(0, 4, (l, n_padded_128, k // 2), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
    a_ref = a_ref.view(torch.float4_e2m1fn_x2)
    b_ref = b_ref.view(torch.float4_e2m1fn_x2)

    c_ref = torch.randn((l, m, n), dtype=torch.float16, device="cuda").permute(1, 2, 0)

    sf_k = ceil_div(k, SF_VEC_SIZE)
    sfa_ref_cpu, sfa_permuted = create_scale_factor_tensors(l, m, sf_k, rand_range=(0, 3))
    sfb_ref_cpu, sfb_permuted = create_scale_factor_tensors(l, n_padded_128, sf_k, rand_range=(0, 3))

    return (a_ref, b_ref, sfa_ref_cpu.to("cuda"), sfb_ref_cpu.to("cuda"), sfa_permuted, sfb_permuted, c_ref)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
