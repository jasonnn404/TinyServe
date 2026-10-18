from collections import deque


class KVCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size

        self.free_blocks = deque(range(num_blocks))
        self.block_tables = {}

    def allocate_for_request(self, request_id, num_tokens):
        num_blocks_needed = (
            num_tokens + self.block_size - 1
        ) // self.block_size

        if num_blocks_needed > len(self.free_blocks):
            raise RuntimeError("Not enough KV cache blocks")

        blocks = [
            self.free_blocks.popleft()
            for _ in range(num_blocks_needed)
        ]

        self.block_tables[request_id] = blocks
        return blocks

    def ensure_capacity(self, request_id, num_tokens):
        blocks = self.block_tables[request_id]

        num_blocks_needed = (
            num_tokens + self.block_size - 1
        ) // self.block_size

        while len(blocks) < num_blocks_needed:
            if not self.free_blocks:
                raise RuntimeError("Not enough KV cache blocks")

            blocks.append(self.free_blocks.popleft())

    def free_request(self, request_id):
        blocks = self.block_tables.pop(request_id)

        for block in blocks:
            self.free_blocks.append(block)

    def get_block_table(self, request_id):
        return self.block_tables[request_id]