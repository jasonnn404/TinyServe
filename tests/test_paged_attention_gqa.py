import math
import torch
from engine.paged_attention import paged_attention_gqa

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

num_q_heads = 16
num_kv_heads = 8
head_dim = 128
block_size = 16
seq_len = 40
block_table = [7, 2, 5]

# Random Q/K/V
q = torch.randn(num_q_heads, head_dim, device=device)
k_ref = torch.randn(num_kv_heads, seq_len, head_dim, device=device)
v_ref = torch.randn_like(k_ref)

# Physical KV storage
k_pool = torch.full((1, 8, 8, 16, 128), 1000.0, device=device)
v_pool = torch.full_like(k_pool, 1000.0)

# Write K/V into physical blocks
for t in range(seq_len):
    physical_block = block_table[t // block_size]
    offset = t % block_size

    k_pool[0, physical_block, :, offset, :] = k_ref[:, t, :]
    v_pool[0, physical_block, :, offset, :] = v_ref[:, t, :]

# Standard GQA reference
group_size = num_q_heads // num_kv_heads

k_expanded = k_ref.repeat_interleave(group_size, dim=0)
v_expanded = v_ref.repeat_interleave(group_size, dim=0)

scores = torch.einsum("hd,htd->ht", q, k_expanded)
scores = scores / math.sqrt(head_dim)

weights = torch.softmax(scores, dim=-1)
expected = torch.einsum("ht,htd->hd", weights, v_expanded)

# Our Paged GQA
actual = paged_attention_gqa(
    q, k_pool, v_pool, block_table,
    layer=0, seq_len=seq_len
)

torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

print("Reference GQA shape:", tuple(expected.shape))
print("Paged GQA shape:", tuple(actual.shape))
print("Max absolute error:", (expected - actual).abs().max().item())
print("Paged GQA correctness: PASS")