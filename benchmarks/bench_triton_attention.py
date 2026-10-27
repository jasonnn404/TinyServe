
import torch
import triton
import triton.testing

from engine.paged_attention import (
    paged_attention_gqa_batch,
    paged_attention_gqa_vectorized,
)
from engine.triton_batched_paged_attention import (
    prepare_paged_metadata,
    triton_paged_attention_batch,
)


@torch.inference_mode()
def bench_attention(batch_size, seq_len):
    device = "cuda"
    block_size = 16
    n_q_heads, n_kv_heads, head_dim = 16, 8, 128
    blocks_per_request = triton.cdiv(seq_len, block_size)

    # Construct non-overlapping physical KV blocks
    n_physical = batch_size * blocks_per_request
    block_tables = [
        list(range(i * blocks_per_request, (i + 1) * blocks_per_request))
        for i in range(batch_size)
    ]
    seq_lens = [seq_len] * batch_size

    q = torch.randn(batch_size, n_q_heads, head_dim, device=device)
    k_pool = torch.randn(
        1, n_physical, n_kv_heads, block_size, head_dim, device=device
    )
    v_pool = torch.randn_like(k_pool)

    tables_gpu, lengths_gpu = prepare_paged_metadata(
        block_tables, seq_lens, block_size, n_physical, device
    )

    def torch_per_request():
        return torch.stack([
            paged_attention_gqa_vectorized(
                q[i], k_pool, v_pool, block_tables[i],
                layer=0, seq_len=seq_lens[i]
            )
            for i in range(batch_size)
        ])

    def torch_batched():
        return paged_attention_gqa_batch(
            q, k_pool, v_pool, block_tables,
            layer=0, seq_lens=seq_lens
        )

    def triton_batched():
        return triton_paged_attention_batch(
            q, k_pool, v_pool, tables_gpu, lengths_gpu, layer=0
        )

    implementations = {
        "PyTorch Per-Request": torch_per_request,
        "PyTorch Batched": torch_batched,
        "Triton Batched": triton_batched,
    }

    # Validate all implementations on identical inputs
    expected = torch_per_request()
    for name, fn in implementations.items():
        torch.testing.assert_close(
            fn(), expected, atol=1e-4, rtol=1e-4
        )

    # Measure attention execution, excluding metadata preparation
    results = {
        name: triton.testing.do_bench(
            fn, warmup=20, rep=100, return_mode="median"
        )
        for name, fn in implementations.items()
    }

    best_torch = min(
        results["PyTorch Per-Request"],
        results["PyTorch Batched"],
    )
    speedup = best_torch / results["Triton Batched"]

    print(
        f"{batch_size:>5} {seq_len:>7} | "
        f"{results['PyTorch Per-Request']:>12.3f} "
        f"{results['PyTorch Batched']:>12.3f} "
        f"{results['Triton Batched']:>12.3f} | "
        f"{speedup:>7.2f}x"
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch Context | Torch/Req(ms) Torch/Batch(ms) Triton(ms) | Speedup")

    for batch_size in [1, 4, 16]:
        for seq_len in [32, 128, 512]:
            bench_attention(batch_size, seq_len)
