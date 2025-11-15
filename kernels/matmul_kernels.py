import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),

        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},  num_warps=8, num_stages=3),
    ],
    key=["B", "IN", "OUT"],
)
@triton.jit
def _forward_int4_fused_kernel(x_q_ptr,
                               w_q_ptr, w_scale_ptr,
                               b_ptr, y_ptr,
                               B, IN, OUT,
                               BLOCK_M: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr,
                               PER_CHANNEL: tl.constexpr,
                               HAS_BIAS: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    acc = tl.full((BLOCK_M, BLOCK_N), 0.0, dtype=tl.float32)

    pid_0_off = (tl.arange(0, BLOCK_M) + pid_0 * BLOCK_M) * OUT
    pid_1_off = tl.arange(0, BLOCK_N) + pid_1 * BLOCK_N
    off = pid_0_off[:, None] + pid_1_off[None, :]
    
    out_mask = ((tl.arange(0, BLOCK_M) + pid_0 * BLOCK_M) < B)[:, None] & \
               (pid_1_off < OUT)[None, :]  

    for k in range(0, IN, BLOCK_K):
        off_x_d0 = (tl.arange(0, BLOCK_M) + pid_0 * BLOCK_M) * IN
        off_x_d1 = (tl.arange(0, BLOCK_K) + k)
        off_x = off_x_d0[:, None] + off_x_d1[None, :]
        mask_x = (off_x_d1 < IN)[None, :] & ((tl.arange(0, BLOCK_M) + pid_0 * BLOCK_M) < B)[:, None]

        packed_IN = (IN + 1) // 2
        global_cols = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)
        out_guard = global_cols < OUT
        safe_cols = tl.where(out_guard, global_cols, 0)
        k_indices = tl.arange(0, BLOCK_K) + k
        row_offsets = safe_cols[None, :] * packed_IN
        byte_cols = (k_indices // 2)[:, None]
        off_w = row_offsets + byte_cols
        mask_w = (k_indices[:, None] < IN) & out_guard[None, :]
        is_high = (k_indices & 1) == 1
        

        x = tl.load(x_q_ptr + off_x, mask_x, 0)
        w_byte = tl.load(w_q_ptr + off_w, mask_w, 0)

        w_u32 = w_byte.to(tl.uint32)
        low = w_u32 & 0xF
        high = (w_u32 >> 4) & 0xF
        sel = is_high[:, None]
        w_nib = tl.where(sel, high, low)
        w_i32 = w_nib.to(tl.int32)
        w_signed_i32 = tl.where(w_i32 < 8, w_i32, w_i32 - 16)

        x_f16 = x.to(tl.float16)
        w_f16 = w_signed_i32.to(tl.float16)
        acc += tl.dot(x_f16, w_f16)
    
        

    if PER_CHANNEL:
        w_scale_mask = pid_1_off < OUT
        w_scale = tl.load(w_scale_ptr + pid_1_off, mask=w_scale_mask)
        alpha = w_scale[None, :].to(tl.float32)
    else:
        w_scale = tl.load(w_scale_ptr)
        alpha = w_scale.to(tl.float32)

    if HAS_BIAS:
        bias_mask = pid_1_off < OUT
        bias = tl.load(b_ptr + pid_1_off, mask=bias_mask, other=0).to(tl.float32)
        acc = acc * alpha + bias[None, :]
    else:
        acc = acc * alpha

   
    tl.store(y_ptr + off, acc.to(tl.float16), out_mask)               

def matmul_int4_fused(x: torch.Tensor,
                      w_q: torch.Tensor,
                      w_scale: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      *, per_channel: bool = True) -> torch.Tensor:

    B, IN = x.shape
    OUT = w_scale.shape[0]

    x_f16 = x.to(torch.float16)
    w_scale_f16 = (w_scale.to(dtype=torch.float16, device=x.device) / 7)
    y = torch.empty((B, OUT), dtype=torch.float16, device=x.device)

    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_M"]),
                     triton.cdiv(OUT, meta["BLOCK_N"]))

    _forward_int4_fused_kernel[grid](x_q_ptr=x_f16,
                               w_q_ptr=w_q, w_scale_ptr=w_scale_f16,
                               b_ptr=bias, y_ptr=y,
                               B=B, IN=IN, OUT=OUT,
                               PER_CHANNEL=per_channel,
                               HAS_BIAS=(bias is not None))

    return y