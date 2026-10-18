import math
import torch


BLOCK_SIZE = 2
NUM_BLOCKS = 4
HEAD_DIM = 2

device = "cuda"

# Shared physical KV memory.
k_pool = torch.zeros(
    NUM_BLOCKS,
    BLOCK_SIZE,
    HEAD_DIM,
    device=device,
)

v_pool = torch.zeros_like(k_pool)


# Each request owns different physical blocks.
block_tables = {
    "A": [3, 1],
    "B": [0],
}

sequence_lengths = {
    "A": 4,
    "B": 2,
}


# -------------------------
# Fake K/V for request A
# -------------------------

a_keys = torch.tensor([
    [1.0, 0.0],   # A0
    [0.0, 1.0],   # A1
    [1.0, 1.0],   # A2
    [-1.0, 0.0],  # A3
], device=device)

a_values = torch.tensor([
    [10.0, 0.0],
    [0.0, 20.0],
    [30.0, 30.0],
    [-10.0, 0.0],
], device=device)


# -------------------------
# Fake K/V for request B
# -------------------------

b_keys = torch.tensor([
    [0.0, 1.0],   # B0
    [1.0, 0.0],   # B1
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
    block_table = block_tables[request_id]
    seq_len = sequence_lengths[request_id]

    keys = []
    values = []

    for physical_block in block_table:
        keys.append(k_pool[physical_block])
        values.append(v_pool[physical_block])

    keys = torch.cat(keys, dim=0)[:seq_len]
    values = torch.cat(values, dim=0)[:seq_len]

    return keys, values


def attention(request_id, q):
    keys, values = gather_kv(request_id)

    scores = q @ keys.T
    scores = scores / math.sqrt(HEAD_DIM)

    weights = torch.softmax(scores, dim=-1)
    output = weights @ values

    return keys, weights, output


write_request("A", a_keys, a_values)
write_request("B", b_keys, b_values)


q_a = torch.tensor([1.0, 0.0], device=device)
q_b = torch.tensor([0.0, 1.0], device=device)

a_k, a_weights, a_output = attention("A", q_a)
b_k, b_weights, b_output = attention("B", q_b)


print("Physical pool:")
print(k_pool)

print("\nA logical K:")
print(a_k)
print("A weights:", a_weights)
print("A output:", a_output)

print("\nB logical K:")
print(b_k)
print("B weights:", b_weights)
print("B output:", b_output)