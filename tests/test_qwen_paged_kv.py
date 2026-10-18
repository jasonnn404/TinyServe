import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
BLOCK_SIZE = 16
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype="auto"
).to(DEVICE).eval()

prompt = "The capital of Canada is Ottawa. " * 4
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
seq_len = inputs["input_ids"].shape[1]

config = model.config

manager = KVCacheManager(
    num_blocks=math.ceil(seq_len / BLOCK_SIZE) + 1,
    block_size=BLOCK_SIZE,
)

pool = KVCachePool(
    num_layers=config.num_hidden_layers,
    num_blocks=manager.num_blocks if hasattr(manager, "num_blocks") else math.ceil(seq_len / BLOCK_SIZE) + 1,
    num_kv_heads=config.num_key_value_heads,
    block_size=BLOCK_SIZE,
    head_dim=config.head_dim,
    dtype=next(model.parameters()).dtype,
    device=DEVICE,
)

cache = PagedKVCache(manager, pool)
cache.add_request("A")


def get_hf_kv(hf_cache, layer):
    if hasattr(hf_cache, "layers"):
        return hf_cache.layers[layer].keys, hf_cache.layers[layer].values
    return hf_cache.key_cache[layer], hf_cache.value_cache[layer]


with torch.inference_mode():
    output = model(**inputs, use_cache=True, logits_to_keep=1)
    hf_cache = output.past_key_values

    for layer in range(config.num_hidden_layers):
        hf_k, hf_v = get_hf_kv(hf_cache, layer)

        # Write the actual Qwen K/V into our paged pool.
        for t in range(seq_len):
            k = hf_k[0, :, t, :]
            v = hf_v[0, :, t, :]
            cache.write_token("A", layer, t, k, v)

        # Read the same K/V back from our pool.
        stored = [cache.read_token("A", layer, t) for t in range(seq_len)]
        read_k = torch.stack([kv[0] for kv in stored], dim=1).unsqueeze(0)
        read_v = torch.stack([kv[1] for kv in stored], dim=1).unsqueeze(0)

        torch.testing.assert_close(read_k, hf_k, atol=0, rtol=0)
        torch.testing.assert_close(read_v, hf_v, atol=0, rtol=0)

        if layer == 0:
            print("HF K shape:", tuple(hf_k.shape))
            print("Paged K shape:", tuple(read_k.shape))

print("Sequence length:", seq_len)
print("Block table:", manager.get_block_table("A"))
print("All 28 layers KV match: PASS")

cache.free_request("A")