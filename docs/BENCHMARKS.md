# TinyServe Benchmark Report

This report documents **observed results**, not projected performance. It records the optimization sequence used to investigate the throughput gap between TinyServe and vLLM on one NVIDIA GPU.

## 1. Hardware and workload

| Setting | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| OS | Ubuntu 24.04 |
| Model | `Qwen/Qwen3-0.6B-Base` |
| Model precision | FP32 for both measured engines |
| HTTP clients | `benchmarks/bench_compare_serving.py` |
| Total requests | 64 |
| HTTP concurrency | 32 |
| Output limit | 32 new tokens/request for the full-serving benchmark |
| Recorded generated tokens | 2,048 per full-serving trial |
| Trials | Five measured trials, preceded by two warmup rounds |
| Summary | Median across measured trials |

Workloads use a small, repeated set of short prompt templates. **This is a short-context, high-concurrency serving experiment**, not a long-context or mixed-production-traffic evaluation.

The servers were measured separately on the same GPU, to avoid cross-process GPU contention. TinyServe's maximum active batch and prefill batch were set to 32, with CUDA Graph decode enabled. The custom engine uses a 16-token KV block size and CUDA Graph shapes `(1, 2, 4, 8, 16, 32)`.

The comparison vLLM launch configuration used during the experiment was:

```bash
vllm serve Qwen/Qwen3-0.6B-Base \
  --host 127.0.0.1 --port 8001 \
  --dtype float32 \
  --max-model-len 512 \
  --max-num-seqs 32 \
  --block-size 16 \
  --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching
```

TinyServe and vLLM have different scheduling, allocation, and kernel implementation details. In particular, matching model, dtype, token count, and HTTP concurrency **does not** make this a controlled microbenchmark of identical kernels.

Exact package revisions, GPU clocks, and run-to-run environmental differences were not recorded in the original experiments; these should be captured for future reproductions rather than guessed.

## 2. Final end-to-end comparison

**Workload:** 64 requests, concurrency 32, 32 output tokens each.

| Metric | TinyServe: original prefill | TinyServe: final | vLLM baseline |
| --- | ---: | ---: | ---: |
| Output throughput (tokens/s) | 1,294.2 | **2,442.4** | 4,880.8 |
| Median TTFT p50 (ms) | 474.5 | **117.8** | 30.7 |
| TTFT p95 (ms) | 477.1 | 120.3 | 34.5 |
| Median end-to-end latency (ms) | 788.1 | **416.9** | 205.2 |
| Median TPOT proxy (ms) | 10.1 | 9.6 | 5.7 |
| Output tokens | 2,048 | 2,048 | 2,048 |

**Observed improvements:**

- Final TinyServe throughput versus its original serving baseline: **1.89×**, or **+88.7%**.
- Median TTFT reduction: **474.5 → 117.8 ms**, approximately **75.2% lower**.
- Median end-to-end request latency reduction: **788.1 → 416.9 ms**, approximately **47.1% lower**.
- vLLM's measured output throughput was approximately **2.00×** the final TinyServe throughput for this workload.

### Individual final throughput trials

| Engine | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Trial 5 | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyServe final (tokens/s) | 2,453.0 | 2,314.7 | 2,442.2 | 2,442.4 | 2,444.8 | **2,442.4** |
| vLLM (tokens/s) | 4,819.6 | 4,874.3 | 4,923.5 | 4,884.7 | 4,880.8 | **4,880.8** |

The trials have some variance, particularly for TinyServe. These are client-observed end-to-end results rather than pure device throughput.

## 3. Prefill optimization experiment

### Short-output experiment: one generated token/request

| Prefill implementation | Output throughput (tokens/s) | TTFT p50 (ms) | E2E p50 (ms) |
| --- | ---: | ---: | ---: |
| Sequential Hugging Face prefill | 68.6 | 461.4 | 464.1 |
| Batched prefill, batch 8 | 222.1 | 140.5 | 141.7 |
| Batched prefill, batch 16 | 268.2 | 115.4 | 116.9 |
| Batched prefill, batch 32 | **295.5** | **104.6** | **105.9** |

All rows: 64 requests, concurrency 32, output limit one token. With one-token output, repeated decode steps are effectively removed, making the test particularly helpful for understanding first-token overhead. The measured single-token throughput improved **4.31×** from sequential to batch-32 prefill.

### Full-serving experiment: 32 generated tokens/request

| Configuration | Output throughput (tokens/s) | TTFT p50 (ms) | E2E p50 (ms) | TPOT proxy (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sequential prefill | 1,294.2 | 474.5 | 788.1 | 10.1 |
| Prefill batch 8 | 2,182.9 | 153.2 | 466.4 | 10.1 |
| Prefill batch 16 | 2,296.1 | 128.5 | 444.3 | 10.1 |
| Prefill batch 32 | 2,366.1 | 117.0 | 429.8 | 10.1 |
| Prefill batch 32 + vectorized argmax | **2,442.4** | 117.8 | 416.9 | 9.6 |

Batching the previously sequential prefill removed a large first-token bottleneck. Increasing the prefill batch beyond eight continued to help, but returned diminishing end-to-end improvements. Vectorizing decode argmax and CPU token readback yielded a further **observed** ~3.2% throughput improvement; a more tightly controlled A/B replay would be required to isolate that small gain from run variance.

### Diagnosis behind the optimization

The original `ContinuousBatchEngine.step()` admitted multiple requests but ran the Hugging Face prefill model **one request at a time**. With 31 newly admitted requests, a measured step took **468.61 ms**; steady-state per-request prefill breakdown was roughly **13 ms HF forward**, **1.4–2.8 ms KV transfer**, and **0.04 ms finalization**. The code held first-token events until the entire engine step returned, further increasing observed TTFT for requests earlier in that step.

The first profiled request showed a **594.69 ms** cold-start HF forward and is excluded from the steady-state characterization. Replacing repeated batch-one forwards with padded batched prefill significantly reduced first-token latency without requiring an entirely new prefill kernel.

## 4. Decode diagnostics

Under batch-32 decode, synchronized wall-time profiling showed steady-state forward calls around **8–9 ms**, with vectorized argmax and token transfer at roughly **0.06 ms**. Ten subsequent batch-32 CUDA Event probes yielded median values of approximately **8.25 ms wall** and **8.23 ms event interval**.

These numbers **do not prove** that all elapsed time is executing GPU kernels: CUDA events can include time during which the stream is idle between recorded events. Device-level kernel time must be inspected independently.

An Nsight Systems `cuda_gpu_kern_sum` from a mixed serving workload found the following percentages of **summed GPU kernel execution time**:

| Kernel category or representative kernel | Percentage of GPU kernel duration |
| --- | ---: |
| CUTLASS SGEMM 64×64 variant | 32.6% |
| CUTLASS SGEMM 128×32 variant | 15.2% |
| MAGMA SGEMM | 12.5% |
| Two common PyTorch elementwise kernel variants | 11.3% combined |
| cuBLASLt split-K reduction | 4.9% |
| Triton batched PagedAttention | 2.6% |
| Triton KV write | 0.4% |

**Interpretation:** this run's aggregated kernel time was heavily dominated by matrix-multiplication-related operations. The summary includes both **prefill and decode** (and potentially startup), so these percentages should **not** be described as a decode-only decomposition. Further optimization of GEMMs, precision, or kernel fusion remains optional future work—not a v1.0 requirement.

## 5. Metrics and caveats

- **Output throughput:** total generated output tokens divided by elapsed client wall time for an entire load-test trial.
- **TTFT (time to first token):** elapsed time until the client receives the first token-bearing streaming event/chunk. TinyServe's token SSE and vLLM's first nonempty streamed text chunk are not identical event semantics.
- **E2E latency:** elapsed time from each HTTP request's client-side start to stream completion.
- **TPOT proxy:** `(end_time - first_token_time) / (generated_tokens - 1)` when more than one token is generated. This includes final stream-completion overhead, so it is not a pure decode kernel latency measurement.
- **Concurrency versus batch size:** 32 concurrent HTTP clients does not guarantee that every device operation has exact GPU batch size 32; actual active batch depends on request arrival and engine scheduling.
- **Output equality:** two engines generating the same count of tokens does **not** prove matching token sequences. Correctness tests within TinyServe compare custom execution paths against references; cross-engine equivalence was not established by the HTTP performance run.
- **Limited distribution:** short, repeated prompt templates; one GPU; one small model; a greedy generation workload. No broad claims about long-context, mixed input/output lengths, many-GPU deployment, or production traffic.

## 6. Reproduce

Run one server at a time on the benchmark GPU. From the project root, with dependencies installed:

```bash
# TinyServe server, terminal A
python -m uvicorn server.api:app --host 127.0.0.1 --port 8000

# TinyServe measurement, terminal B
python -m benchmarks.bench_compare_serving \
  --backend tinyserve --concurrency 32 --requests 64 \
  --max-tokens 32 --trials 5
```

Shut down TinyServe and start vLLM with the configuration from Section 1, then:

```bash
python -m benchmarks.bench_compare_serving \
  --backend vllm --concurrency 32 --requests 64 \
  --max-tokens 32 --trials 5
```

For a future rigorous reproduction, record `git rev-parse HEAD`, PyTorch, Transformers, Triton, CUDA driver and runtime, and vLLM versions; confirm no other process is using the GPU; keep prompt templates, output settings, and the server configuration fixed. The historical values in this report were recorded during development and should not be silently replaced by newer results.
