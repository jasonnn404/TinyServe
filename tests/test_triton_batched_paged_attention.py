import torch

from engine.paged_attention import paged_attention_gqa_vectorized
from engine.triton_batched_paged_attention import (
    prepare_paged_metadata,
    triton_paged_attention_batch,
)

torch.manual_seed(42)

DEVICE = "cuda"
BLOCK_SIZE = 16
D = 128
LAYER = 1

seq_lens = [1, 13, 29, 40]
block_tables = [[7], [5], [9, 2], [1, 4, 8]]

B = len(seq_lens)
q = torch.randn(B, 16, D, device=DEVICE)

k_pool = torch.full((2, 12, 8, BLOCK_SIZE, D), 1000.0, device=DEVICE)
v_pool = torch.full_like(k_pool, 1000.0)

# Fill each request's valid KV slots with real test values
for i, seq_len in enumerate(seq_lens):
    k = torch.randn(8, seq_len, D, device=DEVICE)
    v = torch.randn_like(k)

    for t in range(seq_len):
        physical = block_tables[i][t // BLOCK_SIZE]
        offset = t % BLOCK_SIZE

        k_pool[LAYER, physical, :, offset, :] = k[:, t, :]
        v_pool[LAYER, physical, :, offset, :] = v[:, t, :]

# Reference: previous PyTorch implementation
expected = torch.stack([
    paged_attention_gqa_vectorized(
        q[i], k_pool, v_pool, block_tables[i],
        layer=LAYER, seq_len=seq_lens[i]
    )
    for i in range(B)
])

# Build shared GPU metadata
tables_gpu, lengths_gpu = prepare_paged_metadata(
    block_tables, seq_lens, BLOCK_SIZE,
    n_physical=k_pool.shape[1], device=DEVICE
)

# One Triton launch for the entire batch
actual = triton_paged_attention_batch(
    q, k_pool, v_pool, tables_gpu, lengths_gpu, layer=LAYER
)

torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

error = (actual - expected).abs().max().item()

print("Grid:", (B, q.shape[1]))
print("Expected shape:", tuple(expected.shape))
print("Actual shape:", tuple(actual.shape))
print("Max error:", error)
print("Batched Triton PagedAttention: PASS")