import math
from collections import deque


class Scheduler:
    def __init__(self, max_batch_size=32, total_kv_blocks=None, block_size=16):
        self.max_batch_size = max_batch_size
        self.waiting_queue = deque()

        self.total_kv_blocks = total_kv_blocks
        self.block_size = block_size
        self.reservations = {}

    def blocks_needed(self, request):
        total_tokens = len(request.prompt_ids) + request.max_new_tokens
        return math.ceil(total_tokens / self.block_size)

    def add_request(self, request):
        if self.total_kv_blocks is not None:
            if self.blocks_needed(request) > self.total_kv_blocks:
                raise ValueError("Request exceeds total KV capacity")

        self.waiting_queue.append(request)

    def get_new_requests(self, active_count):
        available_slots = max(0, self.max_batch_size - active_count)
        requests = []

        while self.waiting_queue and len(requests) < available_slots:
            req = self.waiting_queue[0]
            required = self.blocks_needed(req)

            if self.total_kv_blocks is not None:
                available = self.total_kv_blocks - sum(self.reservations.values())

                if required > available:
                    break

                self.reservations[req.request_id] = required

            requests.append(self.waiting_queue.popleft())

        return requests

    def release_request(self, request_id):
        self.reservations.pop(request_id, None)

    @property
    def has_waiting(self):
        return len(self.waiting_queue) > 0