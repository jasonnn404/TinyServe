import torch

from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step
from engine.cuda_graph_dispatcher import CUDAGraphDispatcher


class ContinuousBatchEngine:
    def __init__(
        self, model, scheduler, cache, pool, manager, eos_token_id,
        attention_backend="torch",
    ):
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.pool = pool
        self.manager = manager
        self.eos_token_id = eos_token_id
        self.attention_backend = attention_backend

        self.device = next(model.parameters()).device
        self.active = {}
        self.finished = {}
        self.last_decode_batch = []
        self.prefill_batch_size = 32

        self.graph_dispatcher = None
        if self.attention_backend == "cuda_graph":
            self.graph_dispatcher = CUDAGraphDispatcher(
                model=self.model,
                cache=self.cache,
                pool=self.pool,
                manager=self.manager,
                max_seq_len=512,
                graph_batch_sizes=(1, 2, 4, 8, 16, 32),
            )

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

        # Transfer the prompt's K/V from HF cache to the shared paged pool.
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

        # Prefill predicts the first output token.
        req.mark_prefilled()
        first_token = result.logits[0, -1].argmax().item()
        req.add_token(first_token, self.eos_token_id)

        if req.is_finished:
            self._finish(req)
        else:
            self.active[req.request_id] = req

        return first_token

    @torch.inference_mode()
    def _prefill_batch(self, requests):
        if not requests:
            return []

        for req in requests:
            if not req.prompt_ids or req.max_new_tokens < 1:
                raise ValueError("Invalid prefill request")

        batch_size = len(requests)
        lengths = [len(req.prompt_ids) for req in requests]
        max_len = max(lengths)

        # Left-pad variable-length prompts for a single HF forward pass.
        input_ids = torch.full(
            (batch_size, max_len), self.eos_token_id,
            dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)

        for i, req in enumerate(requests):
            n = lengths[i]
            input_ids[i, -n:] = torch.tensor(
                req.prompt_ids, dtype=torch.long, device=self.device
            )
            attention_mask[i, -n:] = 1

        # Real prompt token positions start at zero, regardless of padding.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)

        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )

        hf_cache = result.past_key_values
        first_tokens = result.logits[:, -1, :].argmax(-1).tolist()

        for req in requests:
            self.cache.add_request(req.request_id)

        # Store only non-padding K/V, using each request's logical positions.
        for layer in range(self.model.config.num_hidden_layers):
            if hasattr(hf_cache, "layers"):
                k = hf_cache.layers[layer].keys
                v = hf_cache.layers[layer].values
            else:
                k = hf_cache.key_cache[layer]
                v = hf_cache.value_cache[layer]

            for i, req in enumerate(requests):
                start = max_len - lengths[i]
                for t in range(lengths[i]):
                    self.cache.write_token(
                        req.request_id, layer, t,
                        k[i, :, start + t, :],
                        v[i, :, start + t, :],
                    )

        events = []
        for i, req in enumerate(requests):
            token = int(first_tokens[i])
            req.mark_prefilled()
            req.add_token(token, self.eos_token_id)
            events.append((req.request_id, "prefill", token))

            if req.is_finished:
                self._finish(req)
            else:
                self.active[req.request_id] = req

        return events

    @torch.inference_mode()
    def step(self):
        events = []

        # 1. Admit requests and prefill in batches.
        new_requests = self.scheduler.get_new_requests(len(self.active))
        for start in range(0, len(new_requests), self.prefill_batch_size):
            batch = new_requests[start:start + self.prefill_batch_size]
            if len(batch) == 1:
                req = batch[0]
                token = self._prefill(req)
                events.append((req.request_id, "prefill", token))
            else:
                events.extend(self._prefill_batch(batch))

        # 2. Select active requests for batched decoding.
        request_ids = list(self.active)
        self.last_decode_batch = request_ids
        if not request_ids:
            return events

        batch_tokens = torch.tensor(
            [[self.active[r].generated_ids[-1]] for r in request_ids],
            dtype=torch.long, device=self.device,
        )
        positions = [
            len(self.active[r].prompt_ids) + len(self.active[r].generated_ids) - 1
            for r in request_ids
        ]

        # 3. Decode with the selected backend.
        if self.attention_backend == "cuda_graph":
            logits = self.graph_dispatcher.decode(
                batch_tokens, request_ids, positions
            )
        else:
            logits = qwen_paged_decode_batch_step(
                self.model, batch_tokens, self.cache,
                self.pool, self.manager, request_ids, positions,
                attention_backend=self.attention_backend,
            )

        # 4. Select all next tokens in one GPU operation and update requests.
        next_tokens = logits.argmax(dim=-1).tolist()
        for i, request_id in enumerate(request_ids):
            req = self.active[request_id]
            token = int(next_tokens[i])
            req.add_token(token, self.eos_token_id)
            events.append((request_id, "decode", token))

            if req.is_finished:
                self._finish(req)

        # Required for the worker and tests to observe this decode step.
        return events

    @property
    def has_work(self):
        return self.scheduler.has_waiting or bool(self.active)
