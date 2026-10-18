import math
import torch


BLOCK_SIZE = 2
NUM_BLOCKS = 4
HEAD_DIM = 2

device = "cuda"

k_pool = torch.zeros(
    NUM_BLOCKS,
    BLOCK_SIZE,
    HEAD_DIM,
    device=device,
)

v_pool = torch.zeros_like(k_pool)


block_tables = {
    "A": [3, 1],
    "B": [0],
}

sequence_lengths = {
    "A": 4,
    "B": 2,
}


a_keys = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [-1.0, 0.0],
], device=device)

a_values = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
    [30.0, 30.0],
    [-10.0, 0.0],
], device=device)


b_keys = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
], device=device)

b_values = torch.tensor([
    [0.0, 100.0],
    [50.0, 0.0],
], device=device)


def write_request(request_id, keys, values):
    block_table = block_tables[request_id]

    for token_index in range(len(keys)):
        logical_block = token_index // BLOCK_SIZE
        offset = token_index % BLOCK_SIZE
        physical_block = block_table[logical_block]

        k_pool[physical_block, offset] = keys[token_index]
        v_pool[physical_block, offset] = values[token_index]


def gather_kv(request_id):
    blocks = block_tables[request_id]
    seq_len = sequence_lengths[request_id]

    keys = torch.cat(
        [k_pool[block] for block in blocks],
        dim=0,
    )[:seq_len]

    values = torch.cat(
        [v_pool[block] for block in blocks],
        dim=0,
    )[:seq_len]

    return keys, values


def ragged_paged_attention(q_batch, request_ids):
    outputs = []

    for i, request_id in enumerate(request_ids):
        q = q_batch[i]

        keys, values = gather_kv(request_id)

        scores = q @ keys.T
        scores = scores / math.sqrt(HEAD_DIM)

        weights = torch.softmax(scores, dim=-1)
        output = weights @ values

        outputs.append(output)

    return torch.stack(outputs)


write_request("A", a_keys, a_values)
write_request("B", b_keys, b_values)


q_batch = torch.tensor([
    [1.0, 0.0],  # Q_A
    [0.0, 1.0],  # Q_B
], device=device)

request_ids = ["A", "B"]

outputs = ragged_paged_attention(
    q_batch,
    request_ids,
)

print("Q batch:")
print(q_batch)

print("\nOutput batch:")
print(outputs)