from engine.request import Request
from engine.scheduler import Scheduler


def make_request(request_id):
    return Request(
        request_id=request_id,
        prompt=f"Prompt {request_id}",
        prompt_ids=[1, 2, 3],
        max_new_tokens=10,
    )


scheduler = Scheduler(max_batch_size=3)
active = []

# Submit 5 requests
for i in range(5):
    scheduler.add_request(make_request(i))

# First admission
new_requests = scheduler.get_new_requests(len(active))
active.extend(new_requests)

assert [r.request_id for r in active] == [0, 1, 2]
assert len(scheduler.waiting_queue) == 2

print("Initial active:", [r.request_id for r in active])

# Simulate request 0 finishing
active.pop(0)

# Admit next waiting request
new_requests = scheduler.get_new_requests(len(active))
active.extend(new_requests)

assert [r.request_id for r in active] == [1, 2, 3]
assert len(scheduler.waiting_queue) == 1

print("After first completion:", [r.request_id for r in active])

# Simulate another completion
active.pop(0)

new_requests = scheduler.get_new_requests(len(active))
active.extend(new_requests)

assert [r.request_id for r in active] == [2, 3, 4]
assert not scheduler.has_waiting

print("Final active:", [r.request_id for r in active])
print("Scheduler admission: PASS")