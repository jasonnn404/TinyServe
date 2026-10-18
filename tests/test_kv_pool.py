import torch


NUM_BLOCKS = 8
BLOCK_SIZE = 4
NUM_KV_HEADS = 1
HEAD_DIM = 1

device = "cuda"

k_pool = torch.zeros(
    NUM_BLOCKS,
    NUM_KV_HEADS,
    BLOCK_SIZE,
    HEAD_DIM,
    device=device,
)

block_table = [5, 2]


def get_cache_location(token_index):
    logical_block = token_index // BLOCK_SIZE
    offset = token_index % BLOCK_SIZE
    physical_block = block_table[logical_block]

    return physical_block, offset


# Write K for token 0~5.
for token_index in range(6):
    physical_block, offset = get_cache_location(token_index)

    # Easy-to-recognize fake K value.
    k = torch.tensor(
        [[float(token_index)]],
        device=device,
    )

    k_pool[physical_block, :, offset, :] = k


# Read them back in logical token order.
recovered = []

for token_index in range(6):
    physical_block, offset = get_cache_location(token_index)

    k = k_pool[
        physical_block,
        :,
        offset,
        :,
    ]

    recovered.append(k.item())


print("Block table:", block_table)

print("\nPhysical block 5:")
print(k_pool[5, 0, :, 0])

print("\nPhysical block 2:")
print(k_pool[2, 0, :, 0])

print("\nRecovered logical sequence:")
print(recovered)