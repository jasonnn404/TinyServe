from engine.request import Request
from engine.scheduler import Scheduler


def make_request(request_id, prompt_len, max_new_tokens):
    return Request(
        request_id=request_id,
        prompt=f"Prompt {request_id}",
        prompt_ids=[1] * prompt_len,
        max_new_tokens=max_new_tokens,
    )


scheduler = Scheduler(
    max_batch_size=3,
    total_kv_blocks=10,
    block_size=16,
)

a = make_request("A", 20, 30)  # 4 blocks
b = make_request("B", 50, 20)  # 5 blocks
c = make_request("C", 16, 16)  # 2 blocks

for req in [a, b, c]:
    scheduler.add_request(req)

# A and B can be admitted, C cannot.
admitted = scheduler.get_new_requests(active_count=0)

assert [r.request_id for r in admitted] == ["A", "B"]
assert sum(scheduler.reservations.values()) == 9
assert scheduler.has_waiting

print("Initial admitted:", [r.request_id for r in admitted])
print("Reserved blocks:", scheduler.reservations)

# Simulate A finishing and releasing its reservation.
scheduler.release_request("A")

# B is still active; C can now join.
admitted = scheduler.get_new_requests(active_count=1)

assert [r.request_id for r in admitted] == ["C"]
assert sum(scheduler.reservations.values()) == 7
assert not scheduler.has_waiting

print("After A finishes:", [r.request_id for r in admitted])
print("Reserved blocks:", scheduler.reservations)
print("Memory-Aware Scheduling: PASS")