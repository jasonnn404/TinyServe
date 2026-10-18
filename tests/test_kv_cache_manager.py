from engine.kv_cache_manager import KVCacheManager


manager = KVCacheManager(
    num_blocks=8,
    block_size=4,
)

manager.allocate_for_request(
    request_id=0,
    num_tokens=6,
)

manager.allocate_for_request(
    request_id=1,
    num_tokens=3,
)

print("A:", manager.get_block_table(0))
print("B:", manager.get_block_table(1))
print("Free:", list(manager.free_blocks))

manager.ensure_capacity(
    request_id=0,
    num_tokens=9,
)

print("\nAfter A grows:")
print("A:", manager.get_block_table(0))
print("Free:", list(manager.free_blocks))

manager.free_request(0)

print("\nAfter A finishes:")
print("Free:", list(manager.free_blocks))