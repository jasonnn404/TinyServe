import math
import torch

from engine.triton_single_block_attention import triton_single_block_attention

torch.manual_seed(42)

device = "cuda"
D = 128
BLOCK = 16

q = torch.randn(D, device=device)
k = torch.randn(BLOCK, D, device=device)
v = torch.randn(BLOCK, D, device=device)

for valid in [16, 13, 1]:
    # Invalid slots deliberately contain bad values
    test_k = k.clone()
    test_v = v.clone()
    test_k[valid:] = 1000.0
    test_v[valid:] = 1000.0

    scores = (test_k[:valid] @ q) / math.sqrt(D)
    expected = torch.softmax(scores, dim=0) @ test_v[:valid]

    actual = triton_single_block_attention(
        q, test_k, test_v, valid
    )

    torch.testing.assert_close(
        actual, expected, atol=1e-4, rtol=1e-4
    )

    error = (actual - expected).abs().max().item()
    print(f"Valid tokens={valid} | Max error={error:.2e}")

print("Triton Single-Block Attention: PASS")