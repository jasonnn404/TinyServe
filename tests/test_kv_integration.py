import torch
from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool

manager = KVCacheManager(num_blocks=4, block_size=4)

pool = KVCachePool(
    num_layers=2,
    num_blocks=4,
    num_kv_heads=2,
    block_size=4,
    head_dim=3,
    dtype=torch.float32,
    device="cuda",
)

# A needs space for 6 tokens.
manager.allocate_for_request("A", num_tokens=6)

token_index = 5
layer = 0

# Translate logical position into physical address.
block_table = manager.get_block_table("A")
logical_block = token_index // manager.block_size
offset = token_index % manager.block_size
physical_block = block_table[logical_block]

# Fake K/V for one token at one layer.
k = torch.ones(2, 3, device="cuda")
v = torch.full((2, 3), 5.0, device="cuda")

pool.write(layer, physical_block, offset, k, v)

stored_k, stored_v = pool.read(layer, physical_block, offset)

assert torch.equal(stored_k, k)
assert torch.equal(stored_v, v)

print("Block table:", block_table)
print("Physical block:", physical_block)
print("Offset:", offset)
print("KV read/write: PASS")