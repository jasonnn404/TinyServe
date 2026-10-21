import math
import torch

from engine.paged_attention import (
    paged_attention_gqa,
    paged_attention_gqa_vectorized,
)

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

num_q_heads = 16
num_kv_heads = 8
head_dim = 128
block_size = 16
seq_len = 40
block_table = [7, 2, 5]

q = torch.randn(num_q_heads, head_dim, device=device)
k_ref = torch.randn(num_kv_heads, seq_len, head_dim, device=device)
v_ref = torch.randn_like(k_ref)

# Deliberately fill unused storage with invalid values
k_pool = torch.full((1, 8, 8, 16, 128), 1000.0, device=device)
v_pool = torch.full_like(k_pool, 1000.0)

for t in range(seq_len):
    block = block_table[t // block_size]
    offset = t % block_size
    k_pool[0, block, :, offset, :] = k_ref[:, t, :]
    v_pool[0, block, :, offset, :] = v_ref[:, t, :]

# Original implementation
expected = paged_attention_gqa(
    q, k_pool, v_pool, block_table, layer=0, seq_len=seq_len
)

# New vectorized implementation
actual = paged_attention_gqa_vectorized(
    q, k_pool, v_pool, block_table, layer=0, seq_len=seq_len
)

torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

print("Reference shape:", tuple(expected.shape))
print("Vectorized shape:", tuple(actual.shape))
print("Max error:", (actual - expected).abs().max().item())
print("Vectorized PagedAttention: PASS")