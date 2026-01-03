import torch

SF_VEC_SIZE = 16


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def create_scale_factor_tensors(l, mn, sf_k, rand_range=(0, 4), dtype_init=torch.int8):
    """
    Create scale factor tensors for both reference and custom kernel layouts.

    Args:
        l: batch size
        mn: M or N dimension size
        sf_k: scale factor K dimension (k // SF_VEC_SIZE)
        rand_range: tuple of (low, high) for random int generation
        dtype_init: initial dtype for random tensor

    Returns:
        (ref_tensor_cpu, permuted_tensor_gpu)
    """
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    ref_f8_random_int = torch.randint(rand_range[0], rand_range[1], ref_shape, dtype=dtype_init, device='cuda')
    ref_f8_torch_tensor = ref_f8_random_int.to(dtype=torch.float8_e4m3fn)
    ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    rand_int_tensor = torch.randint(rand_range[0], rand_range[1], mma_shape, dtype=dtype_init, device='cuda')
    reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

    i_idx = torch.arange(mn, device='cuda')
    j_idx = torch.arange(sf_k, device='cuda')
    b_idx = torch.arange(l, device='cuda')
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k

    reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

    return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor


def create_scale_factor_tensors_fp32(l, mn, sf_k):
    """
    Create scale factor tensors using fp32 random values in [0,1).
    Used by dual_gemm which needs different random distribution.
    """
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    ref_f8_random_fp32 = torch.rand(ref_shape, dtype=torch.float32, device='cuda')
    ref_f8_torch_tensor = ref_f8_random_fp32.to(dtype=torch.float8_e4m3fn)
    ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    rand_int_tensor = torch.empty(mma_shape, dtype=torch.int8, device='cuda')
    reordered_f8_torch_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

    i_idx = torch.arange(mn, device='cuda')
    j_idx = torch.arange(sf_k, device='cuda')
    b_idx = torch.arange(l, device='cuda')
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k

    reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

    return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor
