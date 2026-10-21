import torch
from engine.paged_attention import (
    paged_attention_gqa_vectorized,
    paged_attention_gqa_batch,
)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

seq_lens = [29, 12, 40]
block_tables = [[7, 2], [5], [1, 4, 8]]
block_size = 16

q = torch.randn(3, 16, 128, device=device)
k_pool = torch.full((1, 10, 8, 16, 128), 1000.0, device=device)
v_pool = torch.full_like(k_pool, 1000.0)

for i, seq_len in enumerate(seq_lens):
    k = torch.randn(8, seq_len, 128, device=device)
    v = torch.randn_like(k)

    for t in range(seq_len):
        physical = block_tables[i][t // block_size]
        offset = t % block_size

        k_pool[0, physical, :, offset, :] = k[:, t, :]
        v_pool[0, physical, :, offset, :] = v[:, t, :]

# Previous vectorized implementation as reference
expected = torch.stack([
    paged_attention_gqa_vectorized(
        q[i], k_pool, v_pool, block_tables[i],
        layer=0, seq_len=seq_lens[i]
    )
    for i in range(3)
])

# New batched implementation
actual = paged_attention_gqa_batch(
    q, k_pool, v_pool, block_tables,
    layer=0, seq_lens=seq_lens
)

torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

print("Expected shape:", tuple(expected.shape))
print("Actual shape:", tuple(actual.shape))
print("Max error:", (actual - expected).abs().max().item())
print("Batched PagedAttention: PASS")