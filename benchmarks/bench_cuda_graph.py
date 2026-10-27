
import statistics
import time

import torch
from transformers import AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BATCH_SIZE = 4
WARMUP = 5
REPEATS = 30


def measure(fn, repeats=REPEATS):
    latencies = []

    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()

        fn()
        torch.cuda.synchronize()

        latencies.append((time.perf_counter() - start) * 1000)

    return statistics.median(latencies)


@torch.inference_mode()
def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()

    # Fixed GPU input address
    static_tokens = torch.randint(
        0, model.config.vocab_size,
        (BATCH_SIZE, 1),
        device=DEVICE,
        dtype=torch.long,
    )

    def forward():
        return model(
            input_ids=static_tokens,
            use_cache=False,
            logits_to_keep=1,
        ).logits

    # 1. Warm up on a separate CUDA stream
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(WARMUP):
            forward()

    torch.cuda.current_stream().wait_stream(stream)

    # 2. Capture the GPU operations
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        graph_logits = forward()

    # 3. Test correctness using the original input
    graph.replay()
    torch.cuda.synchronize()

    eager_logits = forward()
    torch.testing.assert_close(
        graph_logits, eager_logits, atol=1e-3, rtol=1e-3
    )

    # 4. Modify data without changing the input address
    static_tokens.fill_(123)

    graph.replay()
    torch.cuda.synchronize()

    eager_logits = forward()
    torch.testing.assert_close(
        graph_logits, eager_logits, atol=1e-3, rtol=1e-3
    )

    print("CUDA Graph correctness: PASS")
    print("Static input address:", static_tokens.data_ptr())

    # 5. Warm up both execution paths
    for _ in range(WARMUP):
        forward()
        graph.replay()

    torch.cuda.synchronize()

    # 6. Compare CPU-to-GPU execution latency
    eager_ms = measure(forward)
    graph_ms = measure(graph.replay)

    print(f"\nBatch size: {BATCH_SIZE}")
    print(f"Eager latency:      {eager_ms:.3f} ms")
    print(f"CUDA Graph latency: {graph_ms:.3f} ms")
    print(f"Speedup:            {eager_ms / graph_ms:.2f}x")


if __name__ == "__main__":
    main()
