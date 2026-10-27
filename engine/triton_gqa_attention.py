import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_kernel(
    Q, K_POOL, V_POOL, BLOCK_TABLE, OUT,
    SEQ_LEN, NUM_BLOCKS, LAYER,
    N_PHYSICAL: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Each Triton program handles one query head
    q_head = tl.program_id(0)
    kv_head = q_head // GROUP_SIZE

    t = tl.arange(0, BLOCK_SIZE)
    d = tl.arange(0, D)

    q = tl.load(Q + q_head * D + d)

    # Online Softmax state
    m = -float("inf")
    l = 0.0
    o = tl.full((D,), 0.0, tl.float32)

    for logical_block in range(NUM_BLOCKS):
        physical = tl.load(BLOCK_TABLE + logical_block)

        # Locate this layer's physical KV block and KV head
        base = (
            (LAYER * N_PHYSICAL + physical)
            * N_KV_HEADS + kv_head
        ) * BLOCK_SIZE * D

        offsets = base + t[:, None] * D + d[None, :]
        valid = logical_block * BLOCK_SIZE + t < SEQ_LEN

        k = tl.load(K_POOL + offsets, mask=valid[:, None], other=0.0)
        v = tl.load(V_POOL + offsets, mask=valid[:, None], other=0.0)

        # Attention scores
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        # Online Softmax
        new_m = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - new_m)
        weights = tl.exp(scores - new_m)

        l = alpha * l + tl.sum(weights, axis=0)
        o = alpha * o + tl.sum(weights[:, None] * v, axis=0)
        m = new_m

    # Each program writes its own query head's output
    tl.store(OUT + q_head * D + d, o / l)


def triton_paged_attention_gqa(
    q, k_pool, v_pool, block_table, layer, seq_len
):
    """
    q: [num_q_heads, head_dim]
    returns: [num_q_heads, head_dim]
    """
    n_layers, n_physical, n_kv_heads, block_size, head_dim = k_pool.shape
    n_q_heads = q.shape[0]

    if q.ndim != 2 or q.shape[1] != head_dim:
        raise ValueError("Invalid query shape")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("Invalid GQA head configuration")
    if not (0 <= layer < n_layers):
        raise ValueError("Invalid layer")
    if not 1 <= seq_len <= block_table.numel() * block_size:
        raise ValueError("Invalid sequence length")
    if v_pool.shape != k_pool.shape:
        raise ValueError("K/V pool shape mismatch")

    tensors = (q, k_pool, v_pool, block_table)
    if not all(x.is_cuda and x.is_contiguous() for x in tensors):
        raise ValueError("Expected contiguous CUDA tensors")
    if len({x.device for x in tensors}) != 1:
        raise ValueError("All tensors must share one GPU")
    if any(x.dtype != torch.float32 for x in (q, k_pool, v_pool)):
        raise ValueError("Expected FP32 tensors")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("Block table must contain integers")
    if triton.next_power_of_2(block_size) != block_size:
        raise ValueError("Block size must be a power of two")
    if triton.next_power_of_2(head_dim) != head_dim:
        raise ValueError("Head dimension must be a power of two")

    output = torch.empty_like(q)

    _gqa_kernel[(n_q_heads,)](
        q, k_pool, v_pool, block_table, output,
        seq_len, triton.cdiv(seq_len, block_size), layer,
        N_PHYSICAL=n_physical,
        N_KV_HEADS=n_kv_heads,
        GROUP_SIZE=n_q_heads // n_kv_heads,
        BLOCK_SIZE=block_size,
        D=head_dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
    )

    return output