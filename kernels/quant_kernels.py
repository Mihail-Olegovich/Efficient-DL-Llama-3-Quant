import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(x_ptr, output_ptr, output_maxs, n_elements, BLOCK_SIZE: tl.constexpr, P2: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * n_elements
    row_start_ptr = x_ptr + block_start

    idx = tl.arange(0, P2 // 2)
    off_even = 2 * idx
    off_odd = 2 * idx + 1

    mask_even = off_even < n_elements
    mask_odd = off_odd < n_elements

    x_even = tl.load(row_start_ptr + off_even, mask=mask_even, other=0.0)
    x_odd = tl.load(row_start_ptr + off_odd, mask=mask_odd, other=0.0)

    absmax_even = tl.max(tl.abs(x_even))
    absmax_odd = tl.max(tl.abs(x_odd))
    absmax = tl.maximum(absmax_even, absmax_odd)

    scale = tl.where(absmax == 0, 0.0, 7.0 / absmax)

    s_even = x_even * scale
    s_odd = x_odd * scale

    q_even = tl.where(s_even >= 0, s_even + 0.5, s_even - 0.5).to(tl.int8).to(tl.uint8) & 0xF
    q_odd = tl.where(s_odd >= 0, s_odd + 0.5, s_odd - 0.5).to(tl.int8).to(tl.uint8) & 0xF

    packed = (q_odd << 4) | q_even

    packed_block_start = pid * ((n_elements + 1) // 2)
    packed_mask = idx < ((n_elements + 1) // 2)

    tl.store(output_ptr + packed_block_start + idx, packed, mask=packed_mask)
    tl.store(output_maxs + pid, absmax)


def quantize_rowwise(x: torch.Tensor):
    N = x.shape[0]
    M = x.shape[1]

    out_cols = (M + 1) // 2

    output_tensor = torch.empty((N, out_cols), dtype=torch.uint8, device=x.device)

    output_maxs = torch.empty(N, dtype=torch.float16, device=x.device)

    P2 = 2 ** int(torch.ceil(torch.log2(torch.tensor(M, dtype=torch.float16))))

    grid = lambda meta: (N,)
    _quantize_rowwise[grid](x_ptr=x, output_ptr=output_tensor, output_maxs=output_maxs, n_elements=M, P2=P2)

    return output_tensor, output_maxs

