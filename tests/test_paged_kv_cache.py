import torch

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache

manager = KVCacheManager(num_blocks=8, block_size=4)

pool = KVCachePool(
    num_layers=2,
    num_blocks=8,
    num_kv_heads=2,
    block_size=4,
    head_dim=3,
    dtype=torch.float32,
    device="cuda",
)

cache = PagedKVCache(manager, pool)
cache.add_request("A")

for token_index in range(9):
    for layer in range(2):
        k = torch.full((2, 3), float(token_index + 10 * layer), device="cuda")
        v = k + 100
        cache.write_token("A", layer, token_index, k, v)

    if token_index in (5, 7, 8):
        print(f"After token {token_index}: {manager.get_block_table('A')}")

k, v = cache.read_token("A", layer=1, token_index=8)

assert torch.all(k == 18)
assert torch.all(v == 118)

print("KV cache integration: PASS")

cache.free_request("A")
print("Free blocks:", len(manager.free_blocks))