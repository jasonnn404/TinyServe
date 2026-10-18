class PagedKVCache:
    def __init__(self, manager, pool):
        self.manager = manager
        self.pool = pool

    def add_request(self, request_id):
        self.manager.allocate_for_request(request_id, num_tokens=0)

    def write_token(self, request_id, layer, token_index, k, v):
        # Make sure enough physical blocks exist.
        self.manager.ensure_capacity(request_id, token_index + 1)

        # Find the physical address.
        block_size = self.manager.block_size
        logical_block = token_index // block_size
        offset = token_index % block_size

        block_table = self.manager.get_block_table(request_id)
        physical_block = block_table[logical_block]

        # Store K/V in GPU memory.
        self.pool.write(layer, physical_block, offset, k, v)

    def read_token(self, request_id, layer, token_index):
        block_size = self.manager.block_size
        logical_block = token_index // block_size
        offset = token_index % block_size

        block_table = self.manager.get_block_table(request_id)
        physical_block = block_table[logical_block]

        return self.pool.read(layer, physical_block, offset)

    def free_request(self, request_id):
        self.manager.free_request(request_id)