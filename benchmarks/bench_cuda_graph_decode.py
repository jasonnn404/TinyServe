
import math
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_cuda_graph_runner import QwenCUDAGraphDecoder


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16
WARMUP = 5
STEPS = 30
TRIALS = 3
BATCH_SIZES = [1, 4, 16]


@torch.inference_mode()
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).to(DEVICE).eval()

    inputs = tokenizer(
        "The capital of Canada is Ottawa. " * 4,
        return_tensors="pt"
    ).to(DEVICE)

    prefill = model(**inputs, use_cache=True, logits_to_keep=1)
    first_token = prefill.logits[:, -1].argmax(-1, keepdim=True)
    hf_cache = prefill.past_key_values

    prompt_len = inputs.input_ids.shape[1]
    config = model.config

    kv_layers = []
    for layer in range(config.num_hidden_layers):
        if hasattr(hf_cache, "layers"):
            k = hf_cache.layers[layer].keys
            v = hf_cache.layers[layer].values
        else:
            k = hf_cache.key_cache[layer]
            v = hf_cache.value_cache[layer]
        kv_layers.append((k, v))

    def create_cache(batch_size, request_ids, max_blocks):
        num_blocks = batch_size * max_blocks + 2

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

        for rid in request_ids:
            cache.add_request(rid)

            for layer_idx, (k, v) in enumerate(kv_layers):
                for t in range(prompt_len):
                    cache.write_token(
                        rid, layer_idx, t,
                        k[0, :, t, :], v[0, :, t, :]
                    )

        return manager, pool, cache

    def benchmark(batch_size, mode):
        request_ids = [f"R{i}" for i in range(batch_size)]
        max_blocks = math.ceil(
            (prompt_len + WARMUP + STEPS + 1) / BLOCK_SIZE
        )

        manager, pool, cache = create_cache(
            batch_size, request_ids, max_blocks
        )

        decoder = QwenCUDAGraphDecoder(
            model, pool, batch_size, max_blocks
        )

        tokens = first_token.repeat(batch_size, 1)
        positions = [prompt_len] * batch_size

        # Prepare the initial state
        decoder.prepare(tokens, manager, request_ids, positions)

        if mode == "graph":
            decoder.capture()

            def execute():
                return decoder.replay()
        else:
            # Initialize Triton kernels before timed execution
            decoder._forward()

            def execute():
                return decoder._forward()

        # Warmup with real autoregressive steps
        for _ in range(WARMUP):
            decoder.prepare(tokens, manager, request_ids, positions)
            logits = execute()

            tokens = logits.argmax(-1, keepdim=True)
            positions = [p + 1 for p in positions]

        torch.cuda.synchronize()

        generated = []
        start = time.perf_counter()

        # Timed continuous decode
        for _ in range(STEPS):
            decoder.prepare(tokens, manager, request_ids, positions)
            logits = execute()

            tokens = logits.argmax(-1, keepdim=True)
            generated.append(tokens)
            positions = [p + 1 for p in positions]

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        history = torch.cat(generated, dim=1).cpu()

        for rid in request_ids:
            cache.free_request(rid)

        return {
            "latency_ms": elapsed / STEPS * 1000,
            "throughput": batch_size * STEPS / elapsed,
            "tokens": history,
        }

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_ID)
    print("Precision: FP32")
    print("Prompt length:", prompt_len)
    print("Measured steps:", STEPS)
    print("Trials:", TRIALS)

    print("\n=== Triton Eager vs CUDA Graph Decode ===")
    print(
        f"{'Batch':>5} | {'Eager ms':>10} {'Graph ms':>10} "
        f"{'Speedup':>9} | {'Tokens match':>12}"
    )
    print("-" * 68)

    for batch_size in BATCH_SIZES:
        eager_latencies = []
        graph_latencies = []
        eager_throughputs = []
        graph_throughputs = []
        all_match = True

        for trial in range(TRIALS):
            # Alternate order to reduce systematic measurement bias
            order = ("eager", "graph") if trial % 2 == 0 else (
                "graph", "eager"
            )

            results = {}
            for mode in order:
                results[mode] = benchmark(batch_size, mode)

            eager_latencies.append(results["eager"]["latency_ms"])
            graph_latencies.append(results["graph"]["latency_ms"])
            eager_throughputs.append(results["eager"]["throughput"])
            graph_throughputs.append(results["graph"]["throughput"])

            matches = torch.equal(
                results["eager"]["tokens"],
                results["graph"]["tokens"],
            )
            all_match &= matches

            print(
                f"  Trial {trial + 1}, Batch {batch_size}: "
                f"Eager={results['eager']['latency_ms']:.2f} ms, "
                f"Graph={results['graph']['latency_ms']:.2f} ms, "
                f"Match={matches}"
            )

        eager_ms = statistics.median(eager_latencies)
        graph_ms = statistics.median(graph_latencies)
        speedup = eager_ms / graph_ms

        print(
            f"{batch_size:>5} | {eager_ms:>10.2f} "
            f"{graph_ms:>10.2f} {speedup:>8.2f}x | "
            f"{str(all_match):>12}"
        )
        print(
            f"      Throughput: "
            f"Eager {statistics.median(eager_throughputs):.1f} tok/s | "
            f"Graph {statistics.median(graph_throughputs):.1f} tok/s"
        )

        if not all_match:
            raise AssertionError("Generated tokens differ between modes")


if __name__ == "__main__":
    main()
