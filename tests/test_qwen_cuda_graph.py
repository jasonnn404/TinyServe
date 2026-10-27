
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step
from engine.qwen_cuda_graph_runner import QwenCUDAGraphDecoder


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BATCH_SIZE = 2
BLOCK_SIZE = 16
STEPS = 6


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
    hf_cache = prefill.past_key_values

    prompt_len = inputs.input_ids.shape[1]
    first_token = prefill.logits[:, -1].argmax(-1, keepdim=True)

    config = model.config
    request_ids = [f"R{i}" for i in range(BATCH_SIZE)]

    def get_kv(layer):
        if hasattr(hf_cache, "layers"):
            return hf_cache.layers[layer].keys, hf_cache.layers[layer].values
        return hf_cache.key_cache[layer], hf_cache.value_cache[layer]

    def create_cache():
        blocks_per_request = math.ceil(
            (prompt_len + STEPS + 1) / BLOCK_SIZE
        )
        num_blocks = BATCH_SIZE * blocks_per_request + 2

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

            for layer in range(config.num_hidden_layers):
                k, v = get_kv(layer)

                for t in range(prompt_len):
                    cache.write_token(
                        rid, layer, t,
                        k[0, :, t, :], v[0, :, t, :]
                    )

        return manager, pool, cache

    ref_manager, ref_pool, ref_cache = create_cache()
    graph_manager, graph_pool, graph_cache = create_cache()

    max_blocks = math.ceil((prompt_len + STEPS + 1) / BLOCK_SIZE)

    decoder = QwenCUDAGraphDecoder(
        model, graph_pool,
        batch_size=BATCH_SIZE,
        max_blocks=max_blocks,
    )

    tokens = first_token.repeat(BATCH_SIZE, 1)
    initial_positions = [prompt_len] * BATCH_SIZE

    # Initialize static buffers before capture
    decoder.prepare(tokens, graph_manager, request_ids, initial_positions)
    decoder.capture()

    original_table_address = decoder.tables.data_ptr()

    for step in range(STEPS):
        positions = [prompt_len + step] * BATCH_SIZE

        expected = qwen_paged_decode_batch_step(
            model, tokens, ref_cache, ref_pool, ref_manager,
            request_ids, positions,
            attention_backend="triton",
        )

        decoder.prepare(
            tokens, graph_manager, request_ids, positions
        )
        actual = decoder.replay()

        torch.testing.assert_close(
            actual, expected, atol=1e-3, rtol=1e-3
        )

        assert decoder.tables.data_ptr() == original_table_address

        error = (actual - expected).abs().max().item()
        print(
            f"Step {step + 1} | "
            f"Seq len={positions[0] + 1} | "
            f"Max error={error:.2e}"
        )

        tokens = expected.argmax(-1, keepdim=True)

    print("Qwen CUDA Graph Decode: PASS")

    for rid in request_ids:
        ref_cache.free_request(rid)
        graph_cache.free_request(rid)


if __name__ == "__main__":
    main()
