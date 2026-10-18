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