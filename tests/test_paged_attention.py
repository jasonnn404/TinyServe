import math
import torch
from engine.paged_attention import paged_attention_one_head

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 16
seq_len = 40
head_dim = 128
block_table = [7, 2, 5]

# Create a sample paged KV pool
k_pool = torch.full((1, 8, 8, 16, 128), 1000.0, device=device)
v_pool = torch.full_like(k_pool, 1000.0)

# Generate reference Q/K/V
q = torch.randn(head_dim, device=device)
k_ref = torch.randn(seq_len, head_dim, device=device)
v_ref = torch.randn(seq_len, head_dim, device=device)

# Store K/V in non-contiguous physical blocks
for t in range(seq_len):
    logical_block = t // block_size
    physical_block = block_table[logical_block]
    offset = t % block_size

    k_pool[0, physical_block, 0, offset] = k_ref[t]
    v_pool[0, physical_block, 0, offset] = v_ref[t]

# Standard attention reference
scores = (k_ref @ q) / math.sqrt(head_dim)
weights = torch.softmax(scores, dim=0)
expected = weights @ v_ref

# Our PagedAttention
actual = paged_attention_one_head(
    q, k_pool, v_pool, block_table,
    layer=0, kv_head=0, seq_len=seq_len
)

torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

print("Reference output shape:", tuple(expected.shape))
print("Paged output shape:", tuple(actual.shape))
print("Max absolute error:", (expected - actual).abs().max().item())
print("PagedAttention correctness: PASS")