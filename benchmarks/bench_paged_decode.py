
import math
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16
WARMUP = 5
STEPS = 30
BATCH_SIZES = [1, 2, 4, 8, 16]
ENABLE_PROFILE = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()

inputs = tokenizer(
    "The capital of Canada is Ottawa. " * 4,
    return_tensors="pt"
).to(DEVICE)

seq_len = inputs.input_ids.shape[1]
config = model.config

# One shared Hugging Face prefill
with torch.inference_mode():
    prefill = model(**inputs, use_cache=True, logits_to_keep=1)

first_token = prefill.logits[:, -1].argmax(-1, keepdim=True)
hf_cache = prefill.past_key_values


def get_kv(layer):
    if hasattr(hf_cache, "layers"):
        return hf_cache.layers[layer].keys, hf_cache.layers[layer].values
    return hf_cache.key_cache[layer], hf_cache.value_cache[layer]


@torch.inference_mode()
def benchmark(batch_size, attention_backend):
    # Every backend starts with a fresh, identical KV cache
    blocks_per_request = math.ceil(
        (seq_len + WARMUP + STEPS + 1) / BLOCK_SIZE
    )
    num_blocks = batch_size * blocks_per_request + 1

    manager = KVCacheManager(
        num_blocks=num_blocks, block_size=BLOCK_SIZE
    )
    pool = KVCachePool(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        num_kv_heads=config.num_key_value_heads,
        block_size=BLOCK_SIZE,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device=DEVICE,
    )
    cache = PagedKVCache(manager, pool)

    request_ids = [f"R{i}" for i in range(batch_size)]

    # Initialize each request with the same prefill K/V
    for rid in request_ids:
        cache.add_request(rid)

        for layer in range(config.num_hidden_layers):
            k, v = get_kv(layer)

            for t in range(seq_len):
                cache.write_token(
                    rid, layer, t,
                    k[0, :, t, :], v[0, :, t, :]
                )

    tokens = first_token.repeat(batch_size, 1)
    positions = [seq_len] * batch_size

    def decode_step():
        nonlocal tokens, positions

        logits = qwen_paged_decode_batch_step(
            model, tokens, cache, pool, manager,
            request_ids, positions,
            attention_backend=attention_backend,
        )

        tokens = logits.argmax(-1, keepdim=True)
        positions = [p + 1 for p in positions]

    # Warmup: not timed
    for _ in range(WARMUP):
        decode_step()

    torch.cuda.synchronize()

    # Timed autoregressive decoding
    generated_tokens = []
    start = time.perf_counter()

    for _ in range(STEPS):
        decode_step()
        generated_tokens.append(tokens)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * STEPS
    throughput = total_tokens / elapsed
    per_request = STEPS / elapsed
    latency_ms = elapsed / STEPS * 1000

    # Capture generated token sequence outside the timed region
    token_history = (
        torch.stack(generated_tokens, dim=1)
        .squeeze(-1)
        .cpu()
    )

    # Optional profiler: after timing to avoid measurement distortion
    if ENABLE_PROFILE and batch_size == 4:
        # Cross the KV block boundary and warm up the new specialization
        for _ in range(2):
            decode_step()

        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            decode_step()

        torch.cuda.synchronize()

        print(f"\n=== {attention_backend} CPU Profile ===")
        print(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=15
        ))

        print(f"\n=== {attention_backend} CUDA Profile ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15
        ))

    for rid in request_ids:
        cache.free_request(rid)

    return {
        "throughput": throughput,
        "per_request": per_request,
        "latency_ms": latency_ms,
        "tokens": token_history,
    }


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_ID)
    print("Precision: FP32")
    print("Prompt length:", seq_len)
    print("Warmup steps:", WARMUP)
    print("Measured steps:", STEPS)

    print("\n=== Full Qwen Decode Benchmark ===")
    print(
        f"{'Batch':>5} | {'Torch ms':>10} {'Triton ms':>10} "
        f"{'Speedup':>9} | {'Tokens match':>12}"
    )
    print("-" * 67)

    for batch_size in BATCH_SIZES:
        torch_result = benchmark(batch_size, "torch")
        triton_result = benchmark(batch_size, "triton")

        speedup = (
            torch_result["latency_ms"] / triton_result["latency_ms"]
        )
        tokens_match = torch.equal(
            torch_result["tokens"],
            triton_result["tokens"],
        )

        print(
            f"{batch_size:>5} | "
            f"{torch_result['latency_ms']:>10.2f} "
            f"{triton_result['latency_ms']:>10.2f} "
            f"{speedup:>8.2f}x | "
            f"{str(tokens_match):>12}"
        )

        print(
            f"      Throughput: "
            f"Torch {torch_result['throughput']:.1f} tok/s | "
            f"Triton {triton_result['throughput']:.1f} tok/s"
        )

        if not tokens_match:
            print("      WARNING: Generated tokens differ between backends!")
