import time
import statistics
import torch
from engine.model_runner import ModelRunner

runner = ModelRunner()
model = runner.model
tokenizer = runner.tokenizer

BATCH_SIZES = BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT_LENGTH = 512
DECODE_STEPS = 50
WARMUP_STEPS = 5


def sync():
    torch.cuda.synchronize()


def make_batch(batch_size, length):
    text = (
        "Machine learning systems process large amounts of data "
        "and generate useful representations. "
    ) * 1000

    ids = tokenizer(text, return_tensors="pt").input_ids

    # For now every request uses the same length.
    input_ids = ids[:, :length].repeat(batch_size, 1)

    return input_ids.to(model.device)


@torch.inference_mode()
def benchmark_batch(batch_size):
    input_ids = make_batch(batch_size, PROMPT_LENGTH)
    attention_mask = torch.ones_like(input_ids)

    # ---------- Prefill ----------
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )

    past_key_values = outputs.past_key_values

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    cache_position = torch.tensor(
        [PROMPT_LENGTH],
        device=model.device,
    )

    decode_times = []

    # ---------- Decode ----------
    for step in range(DECODE_STEPS):
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones((batch_size, 1)),
        ], dim=1)

        sync()
        start = time.perf_counter()

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )

        sync()
        elapsed = time.perf_counter() - start

        past_key_values = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        cache_position += 1

        if step >= WARMUP_STEPS:
            decode_times.append(elapsed)

    latency = statistics.median(decode_times)

    # One decode step produces one token PER request.
    throughput = batch_size / latency

    return latency, throughput


print(
    f"{'Batch':<10}"
    f"{'Decode ms':<15}"
    f"{'Throughput tok/s':<20}"
)

print("-" * 45)

for batch_size in BATCH_SIZES:
    try:
        latency, throughput = benchmark_batch(batch_size)

        print(
            f"{batch_size:<10}"
            f"{latency * 1000:<15.2f}"
            f"{throughput:<20.2f}"
        )

    except torch.OutOfMemoryError:
        print(f"{batch_size:<10}{'OOM':<15}{'-':<20}")
        torch.cuda.empty_cache()