import math
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q, K_POOL, V_POOL, BLOCK_TABLE, OUT,
    SEQ_LEN, NUM_BLOCKS, LAYER, KV_HEAD,
    N_PHYSICAL_BLOCKS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    t = tl.arange(0, BLOCK_SIZE)
    d = tl.arange(0, D)

    q = tl.load(Q + d)

    # Running Online Softmax state
    m = -float("inf")
    l = 0.0
    o = tl.full((D,), 0.0, tl.float32)

    for logical_block in range(NUM_BLOCKS):
        # 1. Logical -> Physical block
        physical = tl.load(BLOCK_TABLE + logical_block)

        # 2. Calculate the physical memory offset
        base = (
            (LAYER * N_PHYSICAL_BLOCKS + physical)
            * N_KV_HEADS + KV_HEAD
        ) * BLOCK_SIZE * D

        offsets = base + t[:, None] * D + d[None, :]

        # 3. Ignore invalid token slots
        valid = logical_block * BLOCK_SIZE + t < SEQ_LEN

        k = tl.load(
            K_POOL + offsets,
            mask=valid[:, None],
            other=0.0,
        )
        v = tl.load(
            V_POOL + offsets,
            mask=valid[:, None],
            other=0.0,
        )

        # 4. Attention scores for this block
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        # 5. Online Softmax
        new_m = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - new_m)
        weights = tl.exp(scores - new_m)

        l = alpha * l + tl.sum(weights, axis=0)
        o = alpha * o + tl.sum(weights[:, None] * v, axis=0)
        m = new_m

    # 6. Normalize once after all blocks
    tl.store(OUT + d, o / l)


def triton_paged_attention_one_head(
    q, k_pool, v_pool, block_table, layer, kv_head, seq_len
):
    n_layers, n_physical, n_kv_heads, block_size, head_dim = k_pool.shape

    if not all(x.is_cuda for x in (q, k_pool, v_pool, block_table)):
        raise ValueError("All tensors must be on CUDA")
    if q.dtype != torch.float32 or k_pool.dtype != torch.float32:
        raise ValueError("This reference kernel requires FP32")
    if v_pool.dtype != torch.float32:
        raise ValueError("V pool must be FP32")
    if not all(x.is_contiguous() for x in (q, k_pool, v_pool, block_table)):
        raise ValueError("Tensors must be contiguous")
    if v_pool.shape != k_pool.shape or q.shape != (head_dim,):
        raise ValueError("Invalid Q/K/V shapes")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("Block table must contain integer indices")
    if not (0 <= layer < n_layers and 0 <= kv_head < n_kv_heads):
        raise ValueError("Invalid layer or KV head")
    if not 1 <= seq_len <= block_table.numel() * block_size:
        raise ValueError("Invalid sequence length")
    if triton.next_power_of_2(block_size) != block_size:
        raise ValueError("Block size must be a power of two")
    if triton.next_power_of_2(head_dim) != head_dim:
        raise ValueError("Head dimension must be a power of two")
    if len({q.device, k_pool.device, v_pool.device, block_table.device}) != 1:
        raise ValueError("Tensors must share the same GPU")

    out = torch.empty_like(q)
    num_blocks = triton.cdiv(seq_len, block_size)

    _paged_attention_kernel[(1,)](
        q, k_pool, v_pool, block_table, out,
        seq_len, num_blocks, layer, kv_head,
        N_PHYSICAL_BLOCKS=n_physical,
        N_KV_HEADS=n_kv_heads,
        BLOCK_SIZE=block_size,
        D=head_dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
    )

    return out