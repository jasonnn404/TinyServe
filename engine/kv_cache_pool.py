import torch


class KVCachePool:
    def __init__(
        self,
        num_layers,
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype,
        device,
    ):
        shape = (
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self.k_pool = torch.empty(shape, dtype=dtype, device=device)
        self.v_pool = torch.empty(shape, dtype=dtype, device=device)

    def write(self, layer, physical_block, offset, k, v):
        self.k_pool[layer, physical_block, :, offset, :] = k
        self.v_pool[layer, physical_block, :, offset, :] = v

    def read(self, layer, physical_block, offset):
        k = self.k_pool[layer, physical_block, :, offset, :]
        v = self.v_pool[layer, physical_block, :, offset, :]

        return k, v