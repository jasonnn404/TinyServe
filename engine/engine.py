from engine.model_runner import ModelRunner
from engine.scheduler import Scheduler


class TinyServeEngine:
    def __init__(self, max_batch_size=32):
        self.runner = ModelRunner()
        self.scheduler = Scheduler(max_batch_size=max_batch_size)

        self.batch = None
        self.next_request_id = 0

    def add_request(self, prompt, max_new_tokens=100):
        request = self.runner.create_request(
            request_id=self.next_request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.next_request_id += 1
        self.scheduler.add_request(request)

        return request

    def step(self):
        # 1. Remove requests that finished last step.
        if self.batch is not None:
            self.batch = self.runner.remove_finished(self.batch)

        active_count = (
            0 if self.batch is None
            else len(self.batch.requests)
        )

        # 2. Fill empty batch slots.
        new_requests = self.scheduler.get_new_requests(active_count)

        if new_requests:
            existing_requests = (
                [] if self.batch is None
                else self.batch.requests
            )

            combined_requests = existing_requests + new_requests

            # V1:
            # rebuild the entire batch KV cache.
            self.batch = self.runner.build_batch(combined_requests)

            return

        # 3. No newcomers, just continue decoding.
        if self.batch is not None:
            self.batch = self.runner.decode_step(self.batch)

    @property
    def is_idle(self):
        return self.batch is None and not self.scheduler.has_waiting