import math
import torch
import triton
import triton.language as tl


@triton.jit
def _batched_paged_attention_kernel(
    Q, K_POOL, V_POOL, TABLES, LENGTHS, OUT,
    LAYER,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    N_PHYSICAL: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Each program handles one request and one query head
    request_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // GROUP_SIZE

    t = tl.arange(0, BLOCK_SIZE)
    d = tl.arange(0, D)

    seq_len = tl.load(LENGTHS + request_idx)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    q_offset = (request_idx * N_Q_HEADS + q_head) * D
    q = tl.load(Q + q_offset + d)

    # Independent online softmax state
    m = -float("inf")
    l = 0.0
    o = tl.full((D,), 0.0, tl.float32)

    for logical_block in range(num_blocks):
        # Read this request's physical block ID
        table_offset = request_idx * MAX_BLOCKS + logical_block
        physical = tl.load(TABLES + table_offset)

        # Locate K/V for the current layer and KV head
        base = (
            (LAYER * N_PHYSICAL + physical) * N_KV_HEADS
            + kv_head
        ) * BLOCK_SIZE * D

        offsets = base + t[:, None] * D + d[None, :]
        valid = logical_block * BLOCK_SIZE + t < seq_len

        k = tl.load(K_POOL + offsets, mask=valid[:, None], other=0.0)
        v = tl.load(V_POOL + offsets, mask=valid[:, None], other=0.0)

        # Attention scores
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))

        # Online softmax across physical blocks
        new_m = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - new_m)
        weights = tl.exp(scores - new_m)

        l = alpha * l + tl.sum(weights, axis=0)
        o = alpha * o + tl.sum(weights[:, None] * v, axis=0)
        m = new_m

    # Write the result for this request and query head
    tl.store(OUT + q_offset + d, o / l)


def prepare_paged_metadata(block_tables, seq_lens, block_size, n_physical, device):
    """Prepare GPU metadata once for the whole decode batch."""
    if not block_tables or len(block_tables) != len(seq_lens):
        raise ValueError("Invalid batch metadata")

    max_blocks = max(triton.cdiv(n, block_size) for n in seq_lens)
    rows = []

    for table, seq_len in zip(block_tables, seq_lens):
        if seq_len < 1:
            raise ValueError("Sequence length must be positive")

        needed = triton.cdiv(seq_len, block_size)
        if len(table) < needed:
            raise ValueError("Block table is too short")

        selected = table[:needed]
        if any(not 0 <= p < n_physical for p in selected):
            raise ValueError("Invalid physical block index")

        rows.append(selected + [-1] * (max_blocks - needed))

    return (
        torch.tensor(rows, dtype=torch.int32, device=device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def triton_paged_attention_batch(
    q, k_pool, v_pool, tables_gpu, lengths_gpu, layer
):
    """
    q:           [B, num_q_heads, head_dim]
    tables_gpu:  [B, max_blocks]
    lengths_gpu: [B]
    returns:     [B, num_q_heads, head_dim]
    """
    if q.ndim != 3 or k_pool.ndim != 5:
        raise ValueError("Invalid tensor dimensions")

    B, n_q_heads, head_dim = q.shape
    n_layers, n_physical, n_kv_heads, block_size, pool_dim = k_pool.shape

    if B < 1 or head_dim != pool_dim or v_pool.shape != k_pool.shape:
        raise ValueError("Invalid Q/K/V shapes")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("Invalid GQA head configuration")
    if tables_gpu.ndim != 2 or tables_gpu.shape[0] != B:
        raise ValueError("Invalid block tables")
    if lengths_gpu.shape != (B,) or not 0 <= layer < n_layers:
        raise ValueError("Invalid lengths or layer")

    tensors = (q, k_pool, v_pool, tables_gpu, lengths_gpu)
    if not all(x.is_cuda and x.is_contiguous() for x in tensors):
        raise ValueError("Expected contiguous CUDA tensors")
    if len({x.device for x in tensors}) != 1:
        raise ValueError("Tensors must share the same GPU")
    if any(x.dtype != torch.float32 for x in (q, k_pool, v_pool)):
        raise ValueError("Expected FP32 Q/K/V")
    if tables_gpu.dtype != torch.int32 or lengths_gpu.dtype != torch.int32:
        raise ValueError("Expected int32 metadata")
    if triton.next_power_of_2(block_size) != block_size:
        raise ValueError("Block size must be power of two")
    if triton.next_power_of_2(head_dim) != head_dim:
        raise ValueError("Head dimension must be power of two")

    output = torch.empty_like(q)

    _batched_paged_attention_kernel[(B, n_q_heads)](
        q, k_pool, v_pool, tables_gpu, lengths_gpu, output,
        layer,
        N_Q_HEADS=n_q_heads,
        N_KV_HEADS=n_kv_heads,
        N_PHYSICAL=n_physical,
        MAX_BLOCKS=tables_gpu.shape[1],
        GROUP_SIZE=n_q_heads // n_kv_heads,
        BLOCK_SIZE=block_size,
        D=head_dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
    )

    return output