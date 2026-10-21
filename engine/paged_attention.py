import math
import torch


def paged_attention_one_head(
    q, k_pool, v_pool, block_table, layer, kv_head, seq_len
):
    """
    Reference PagedAttention for one query head.

    q:           [head_dim]
    k_pool:      [layers, blocks, kv_heads, block_size, head_dim]
    v_pool:      same shape
    block_table: logical -> physical block mapping
    """
    block_size = k_pool.shape[3]
    head_dim = q.shape[0]

    if seq_len < 1 or seq_len > len(block_table) * block_size:
        raise ValueError("Invalid sequence length or block table")

    q = q.float()

    # Online Softmax state
    m = torch.tensor(float("-inf"), device=q.device)
    l = torch.zeros((), device=q.device)
    o = torch.zeros(head_dim, device=q.device)

    for logical_block, physical_block in enumerate(block_table):
        start = logical_block * block_size

        if start >= seq_len:
            break

        valid = min(block_size, seq_len - start)

        # Read the current physical block
        k = k_pool[layer, physical_block, kv_head, :valid, :].float()
        v = v_pool[layer, physical_block, kv_head, :valid, :].float()

        # Attention scores for this block
        scores = (k @ q) / math.sqrt(head_dim)

        # Online Softmax update
        new_m = torch.maximum(m, scores.max())
        alpha = torch.exp(m - new_m)
        weights = torch.exp(scores - new_m)

        l = alpha * l + weights.sum()
        o = alpha * o + weights @ v
        m = new_m

    return o / l


def paged_attention_gqa(
    q, k_pool, v_pool, block_table, layer, seq_len
):
    """
    GQA Reference PagedAttention.

    q: [num_q_heads, head_dim]
    returns: [num_q_heads, head_dim]
    """
    num_q_heads = q.shape[0]
    num_kv_heads = k_pool.shape[2]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")

    group_size = num_q_heads // num_kv_heads
    outputs = []

    for q_head in range(num_q_heads):
        kv_head = q_head // group_size

        out = paged_attention_one_head(
            q[q_head], k_pool, v_pool,
            block_table, layer, kv_head, seq_len
        )
        outputs.append(out)

    return torch.stack(outputs, dim=0)

def paged_attention_gqa_vectorized(
    q, k_pool, v_pool, block_table, layer, seq_len
):
    """
    Vectorized GQA with block-wise Online Softmax.

    q: [num_q_heads, head_dim]
    returns: [num_q_heads, head_dim]
    """
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[2]
    block_size = k_pool.shape[3]

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("Invalid GQA head configuration")
    if not 1 <= seq_len <= len(block_table) * block_size:
        raise ValueError("Invalid sequence length")

    group_size = num_q_heads // num_kv_heads

    # Group query heads by their shared KV head
    q = q.float().reshape(num_kv_heads, group_size, head_dim)

    # Online Softmax state for every query head
    m = torch.full(
        (num_kv_heads, group_size),
        -float("inf"), device=q.device
    )
    l = torch.zeros(num_kv_heads, group_size, device=q.device)
    o = torch.zeros(num_kv_heads, group_size, head_dim, device=q.device)

    for logical_block, physical_block in enumerate(block_table):
        start = logical_block * block_size
        if start >= seq_len:
            break

        valid = min(block_size, seq_len - start)

        k = k_pool[layer, physical_block, :, :valid, :].float()
        v = v_pool[layer, physical_block, :, :valid, :].float()

        # Compute scores for all query heads together
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / math.sqrt(head_dim)

        # Vectorized Online Softmax
        new_m = torch.maximum(m, scores.max(dim=-1).values)
        alpha = torch.exp(m - new_m)
        weights = torch.exp(scores - new_m.unsqueeze(-1))

        l = alpha * l + weights.sum(dim=-1)
        o = alpha.unsqueeze(-1) * o + torch.bmm(weights, v)
        m = new_m

    return (o / l.unsqueeze(-1)).reshape(num_q_heads, head_dim)


def paged_attention_gqa_batch(
    q, k_pool, v_pool, block_tables, layer, seq_lens
):
    """
    Batched GQA PagedAttention.

    q:            [B, num_q_heads, head_dim]
    block_tables: list of block tables, one per request
    seq_lens:     valid KV lengths for each request
    returns:      [B, num_q_heads, head_dim]
    """
    B, num_q_heads, head_dim = q.shape
    num_kv_heads = k_pool.shape[2]
    block_size = k_pool.shape[3]

    assert num_q_heads % num_kv_heads == 0
    assert len(block_tables) == len(seq_lens) == B
    assert all(1 <= n <= len(t) * block_size
               for n, t in zip(seq_lens, block_tables))

    group_size = num_q_heads // num_kv_heads
    max_blocks = math.ceil(max(seq_lens) / block_size)

    lengths = torch.tensor(seq_lens, device=q.device)
    offsets = torch.arange(block_size, device=q.device)

    q = q.float().reshape(B * num_kv_heads, group_size, head_dim)

    # One Online Softmax state per request and query head
    m = torch.full(
        (B, num_kv_heads, group_size), -float("inf"), device=q.device
    )
    l = torch.zeros_like(m)
    o = torch.zeros(
        B, num_kv_heads, group_size, head_dim, device=q.device
    )

    for logical_block in range(max_blocks):
        # Map logical blocks to physical blocks for the whole batch
        physical = torch.tensor([
            table[logical_block] if logical_block < len(table) else 0
            for table in block_tables
        ], device=q.device)

        k = k_pool[layer, physical].float()
        v = v_pool[layer, physical].float()

        k = k.reshape(B * num_kv_heads, block_size, head_dim)
        v = v.reshape(B * num_kv_heads, block_size, head_dim)

        # Compute all requests and query heads together
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores.reshape(
            B, num_kv_heads, group_size, block_size
        ) / math.sqrt(head_dim)

        # Exclude padding and missing logical blocks
        valid = logical_block * block_size + offsets[None, :]
        valid = valid < lengths[:, None]
        scores = scores.masked_fill(
            ~valid[:, None, None, :], -float("inf")
        )

        # Batched Online Softmax
        new_m = torch.maximum(m, scores.max(dim=-1).values)
        alpha = torch.exp(m - new_m)
        weights = torch.exp(scores - new_m.unsqueeze(-1))

        l = alpha * l + weights.sum(dim=-1)

        weighted_v = torch.bmm(
            weights.reshape(B * num_kv_heads, group_size, block_size),
            v
        ).reshape(B, num_kv_heads, group_size, head_dim)

        o = alpha.unsqueeze(-1) * o + weighted_v
        m = new_m

    return (o / l.unsqueeze(-1)).reshape(B, num_q_heads, head_dim)