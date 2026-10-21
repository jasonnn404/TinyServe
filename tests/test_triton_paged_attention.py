import math
import torch

from engine.triton_paged_attention import triton_paged_attention_one_head

torch.manual_seed(42)

DEVICE = "cuda"
BLOCK_SIZE = 16
D = 128
LAYER = 1
KV_HEAD = 3

block_table = [7, 2, 5]
table_gpu = torch.tensor(block_table, device=DEVICE, dtype=torch.int32)

k_pool = torch.full((2, 10, 8, BLOCK_SIZE, D), 1000.0, device=DEVICE)
v_pool = torch.full_like(k_pool, 1000.0)

q = torch.randn(D, device=DEVICE)

for seq_len in [1, 13, 16, 17, 29, 32, 40]:
    k_pool.fill_(1000.0)
    v_pool.fill_(1000.0)

    k_ref = torch.randn(seq_len, D, device=DEVICE)
    v_ref = torch.randn(seq_len, D, device=DEVICE)

    # Write reference K/V into non-contiguous physical blocks
    for t in range(seq_len):
        physical = block_table[t // BLOCK_SIZE]
        offset = t % BLOCK_SIZE

        k_pool[LAYER, physical, KV_HEAD, offset] = k_ref[t]
        v_pool[LAYER, physical, KV_HEAD, offset] = v_ref[t]

    # Standard Attention reference
    scores = (k_ref @ q) / math.sqrt(D)
    expected = torch.softmax(scores, dim=0) @ v_ref

    # Our Triton PagedAttention
    actual = triton_paged_attention_one_head(
        q, k_pool, v_pool, table_gpu,
        layer=LAYER, kv_head=KV_HEAD, seq_len=seq_len
    )

    torch.testing.assert_close(
        actual, expected, atol=1e-4, rtol=1e-4
    )

    error = (actual - expected).abs().max().item()
    print(f"Seq len={seq_len:2d} | Max error={error:.2e}")

print("Multi-Block Triton PagedAttention: PASS")