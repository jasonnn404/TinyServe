import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, OUT, VALID,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
):
    # 1. Token and head-dimension indices
    t = tl.arange(0, BLOCK)
    d = tl.arange(0, D)

    # 2. Load Q/K/V from GPU memory
    q = tl.load(Q + d)
    k = tl.load(
        K + t[:, None] * D + d[None, :],
        mask=t[:, None] < VALID, other=0.0
    )
    v = tl.load(
        V + t[:, None] * D + d[None, :],
        mask=t[:, None] < VALID, other=0.0
    )

    # 3. Compute attention scores
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    scores = tl.where(t < VALID, scores, -float("inf"))

    # 4. Stable softmax
    weights = tl.exp(scores - tl.max(scores, axis=0))
    weights = weights / tl.sum(weights, axis=0)

    # 5. Weighted sum of V
    output = tl.sum(weights[:, None] * v, axis=0)

    # 6. Write the final output
    tl.store(OUT + d, output)


def triton_single_block_attention(q, k, v, valid_tokens):
    block_size, head_dim = k.shape

    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float32
    assert q.shape == (head_dim,) and v.shape == k.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert 1 <= valid_tokens <= block_size
    assert triton.next_power_of_2(block_size) == block_size
    assert triton.next_power_of_2(head_dim) == head_dim

    out = torch.empty_like(q)

    _attention_kernel[(1,)](
        q, k, v, out, valid_tokens,
        BLOCK=block_size,
        D=head_dim,
        SCALE=1.0 / math.sqrt(head_dim),
        num_warps=4,
    )

    return out