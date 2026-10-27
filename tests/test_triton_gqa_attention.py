import torch

from engine.paged_attention import paged_attention_gqa_vectorized
from engine.triton_gqa_attention import triton_paged_attention_gqa

torch.manual_seed(42)

DEVICE = "cuda"
BLOCK_SIZE = 16
D = 128
N_Q_HEADS = 16
N_KV_HEADS = 8

block_table = [7, 2, 5]
block_table_gpu = torch.tensor(
    block_table, device=DEVICE, dtype=torch.int32
)

q = torch.randn(N_Q_HEADS, D, device=DEVICE)

k_pool = torch.full(
    (2, 10, N_KV_HEADS, BLOCK_SIZE, D), 1000.0, device=DEVICE
)
v_pool = torch.full_like(k_pool, 1000.0)

for seq_len in [1, 13, 16, 17, 29, 40]:
    k_pool.fill_(1000.0)
    v_pool.fill_(1000.0)

    k_ref = torch.randn(N_KV_HEADS, seq_len, D, device=DEVICE)
    v_ref = torch.randn_like(k_ref)

    for t in range(seq_len):
        physical = block_table[t // BLOCK_SIZE]
        offset = t % BLOCK_SIZE

        k_pool[0, physical, :, offset, :] = k_ref[:, t, :]
        v_pool[0, physical, :, offset, :] = v_ref[:, t, :]

    # PyTorch reference
    expected = paged_attention_gqa_vectorized(
        q, k_pool, v_pool, block_table,
        layer=0, seq_len=seq_len
    )

    # Triton implementation
    actual = triton_paged_attention_gqa(
        q, k_pool, v_pool, block_table_gpu,
        layer=0, seq_len=seq_len
    )

    torch.testing.assert_close(
        actual, expected, atol=1e-4, rtol=1e-4
    )

    error = (actual - expected).abs().max().item()
    print(f"Seq len={seq_len:2d} | Max error={error:.2e}")

print("Triton GQA PagedAttention: PASS")