import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16
WARMUP = 5
STEPS = 30

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

# Prepare one reference prefill
with torch.inference_mode():
    prefill = model(**inputs, use_cache=True, logits_to_keep=1)

first_token = prefill.logits[:, -1].argmax(-1, keepdim=True)
hf_cache = prefill.past_key_values

def get_kv(layer):
    if hasattr(hf_cache, "layers"):
        return hf_cache.layers[layer].keys, hf_cache.layers[layer].values
    return hf_cache.key_cache[layer], hf_cache.value_cache[layer]


@torch.inference_mode()
def benchmark(batch_size):
    blocks_per_request = math.ceil(
        (seq_len + WARMUP + STEPS + 1) / BLOCK_SIZE
    )
    num_blocks = batch_size * blocks_per_request + 1

    manager = KVCacheManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
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

    # Initialize each request's KV Cache
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
            request_ids, positions
        )

        tokens = logits.argmax(-1, keepdim=True)
        positions = [p + 1 for p in positions]

    # Warmup (not timed)
    for _ in range(WARMUP):
        decode_step()

    torch.cuda.synchronize()
    start = time.perf_counter()

    # Timed decode
    for _ in range(STEPS):
        decode_step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * STEPS
    throughput = total_tokens / elapsed
    per_request = STEPS / elapsed
    latency_ms = elapsed / STEPS * 1000

    if False and batch_size == 4:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            decode_step()

        torch.cuda.synchronize()

        print("\n=== CPU Profile ===")
        print(prof.key_averages().table(
            sort_by="cpu_time_total", row_limit=15
        ))

        print("\n=== CUDA Profile ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=15
        ))

    for rid in request_ids:
        cache.free_request(rid)

    return throughput, per_request, latency_ms


for batch_size in [1, 2, 4, 8, 16]:
    total, per_req, latency = benchmark(batch_size)

    print(
        f"Batch {batch_size} | "
        f"Total: {total:.1f} tok/s | "
        f"Per request: {per_req:.1f} tok/s | "
        f"Step latency: {latency:.2f} ms"
    )