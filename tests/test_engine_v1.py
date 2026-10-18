from engine.engine import TinyServeEngine


engine = TinyServeEngine(max_batch_size=2)

a = engine.add_request(
    "Count from one to ten:",
    max_new_tokens=4,
)

b = engine.add_request(
    "The capital of Canada is",
    max_new_tokens=1,
)

c = engine.add_request(
    "Machine learning is",
    max_new_tokens=2,
)


def print_state(step):
    print(f"\n--- Step {step} ---")

    if engine.batch is None:
        print("Active: []")
    else:
        print("Active:", [
            request.request_id
            for request in engine.batch.requests
        ])

    print("Waiting:", [
        request.request_id
        for request in engine.scheduler.waiting_queue
    ])

    for request in [a, b, c]:
        print(
            f"Request {request.request_id}: "
            f"generated={len(request.generated_ids)}, "
            f"status={request.status.name}"
        )


print_state(0)

step = 1

while not engine.is_idle and step <= 10:
    engine.step()
    print_state(step)
    step += 1