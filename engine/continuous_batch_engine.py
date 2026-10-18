import torch

from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step


class ContinuousBatchEngine:
    def __init__(self, model, scheduler, cache, pool, manager, eos_token_id):
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.pool = pool
        self.manager = manager
        self.eos_token_id = eos_token_id

        self.device = next(model.parameters()).device
        self.active = {}
        self.finished = {}
        self.last_decode_batch = []

    def _finish(self, req):
        self.cache.free_request(req.request_id)
        self.scheduler.release_request(req.request_id)

        self.active.pop(req.request_id, None)
        self.finished[req.request_id] = req

    def _prefill(self, req):
        if not req.prompt_ids or req.max_new_tokens < 1:
            raise ValueError("Prompt and max_new_tokens must be non-empty")

        input_ids = torch.tensor(
            [req.prompt_ids], dtype=torch.long, device=self.device
        )
        result = self.model(
            input_ids=input_ids, use_cache=True, logits_to_keep=1
        )

        hf_cache = result.past_key_values
        self.cache.add_request(req.request_id)

        # Transfer all prompt K/V into our shared paged pool.
        for layer in range(self.model.config.num_hidden_layers):
            if hasattr(hf_cache, "layers"):
                k = hf_cache.layers[layer].keys
                v = hf_cache.layers[layer].values
            else:
                k = hf_cache.key_cache[layer]
                v = hf_cache.value_cache[layer]

            for t in range(len(req.prompt_ids)):
                self.cache.write_token(
                    req.request_id, layer, t,
                    k[0, :, t, :], v[0, :, t, :]
                )

        # Prefill already predicts the first generated token.
        req.mark_prefilled()
        first_token = result.logits[0, -1].argmax().item()
        req.add_token(first_token, self.eos_token_id)

        if req.is_finished:
            self._finish(req)
        else:
            self.active[req.request_id] = req

        return first_token

    @torch.inference_mode()
    def step(self):
        events = []

        # 1. Admit and prefill new requests.
        new_requests = self.scheduler.get_new_requests(len(self.active))

        for req in new_requests:
            token = self._prefill(req)
            events.append((req.request_id, "prefill", token))

        # 2. Select active requests for batched decoding.
        request_ids = list(self.active)
        self.last_decode_batch = request_ids

        if not request_ids:
            return events

        batch_tokens = torch.tensor(
            [[self.active[r].generated_ids[-1]] for r in request_ids],
            dtype=torch.long, device=self.device
        )

        positions = [
            len(self.active[r].prompt_ids)
            + len(self.active[r].generated_ids) - 1
            for r in request_ids
        ]

        # 3. Execute one batched decoding forward.
        logits = qwen_paged_decode_batch_step(
            self.model, batch_tokens, self.cache,
            self.pool, self.manager, request_ids, positions
        )

        # 4. Record generated tokens and release finished requests.
        for i, request_id in enumerate(request_ids):
            req = self.active[request_id]
            token = logits[i].argmax().item()

            req.add_token(token, self.eos_token_id)
            events.append((request_id, "decode", token))

            if req.is_finished:
                self._finish(req)

        return events

    @property
    def has_work(self):
        return self.scheduler.has_waiting or bool(self.active)