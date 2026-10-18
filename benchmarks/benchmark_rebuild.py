import time
import torch

from engine.model_runner import ModelRunner


runner = ModelRunner()

ACTIVE_LENGTHS = [128, 512, 1024, 2048]
NEW_PROMPT_LENGTH = 16


def sync():
    torch.cuda.synchronize()


def make_request(request_id, length):
    text = (
        "Machine learning systems can process information efficiently. "
        * 1000
    )

    ids = runner.tokenizer(text, return_tensors="pt").input_ids[0, :length].tolist()

    return runner.create_request(
        request_id=request_id,
        prompt="",
        max_new_tokens=100,
    ), ids


print(
    f"{'Active len':<15}"
    f"{'Decode ms':<15}"
    f"{'Rebuild ms':<15}"
    f"{'Penalty':<10}"
)

print("-" * 55)


for active_length in ACTIVE_LENGTHS:
    active, active_ids = make_request(0, active_length)
    newcomer, newcomer_ids = make_request(1, NEW_PROMPT_LENGTH)

    active.prompt_ids = active_ids
    newcomer.prompt_ids = newcomer_ids

    # Build A once so it has a KV cache and a next token.
    batch = runner.build_batch([active])

    # ---------------- Normal decode ----------------
    sync()
    start = time.perf_counter()

    runner.decode_step(batch)

    sync()
    decode_time = time.perf_counter() - start

    # ---------------- V1 rebuild ----------------
    # A now has its full history in token_ids.
    # C is a brand-new short request.
    sync()
    start = time.perf_counter()

    runner.build_batch([active, newcomer])

    sync()
    rebuild_time = time.perf_counter() - start

    print(
        f"{active_length:<15}"
        f"{decode_time * 1000:<15.2f}"
        f"{rebuild_time * 1000:<15.2f}"
        f"{rebuild_time / decode_time:<10.2f}x"
    )