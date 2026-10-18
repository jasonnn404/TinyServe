import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.request import Request
from engine.batch import BatchState


class ModelRunner:
    def __init__(self, model_name="Qwen/Qwen3-0.6B-Base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()


    @torch.no_grad()
    def generate_naive(self, prompt, max_new_tokens=20):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        for _ in range(max_new_tokens):
            outputs = self.model(input_ids=input_ids)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)


    @torch.no_grad()
    def generate_cached(self, prompt, max_new_tokens=20):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_ids = input_ids
        past_key_values = None

        # First prefill contains positions 0, 1, ..., T-1
        cache_position = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # Only the new token needs to go through the model.
            input_ids = next_token_id

            # But attention still covers the entire history.
            attention_mask = torch.cat([
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))
            ], dim=1)

            # The next token occupies the next position in the cache.
            cache_position = cache_position[-1:] + 1

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


    def create_request(self, request_id, prompt, max_new_tokens=100):
        prompt_ids = self.tokenizer.encode(prompt)

        return Request(
            request_id=request_id,
            prompt=prompt,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )


    @torch.inference_mode()
    def build_batch(self, requests):
        sequences = [request.token_ids for request in requests]

        max_length = max(len(ids) for ids in sequences)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []

        for ids in sequences:
            padding_length = max_length - len(ids)

            input_ids.append(
                [pad_token_id] * padding_length + ids
            )

            attention_masks.append(
                [0] * padding_length + [1] * len(ids)
            )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.model.device,
        )

        attention_mask = torch.tensor(
            attention_masks,
            dtype=torch.long,
            device=self.model.device,
        )

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
        )

        next_token_ids = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        # IMPORTANT:
        # lengths currently stored in KV cache,
        # BEFORE adding the newly predicted token.
        sequence_lengths = attention_mask.sum(dim=1)

        for request, token_id in zip(requests, next_token_ids):
            request.add_token(
                token_id.item(),
                self.tokenizer.eos_token_id,
            )
            

        return BatchState(
            requests=requests,
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
            next_token_ids=next_token_ids,
            sequence_lengths=sequence_lengths,
            cache_position=torch.tensor(
                [max_length],
                dtype=torch.long,
                device=self.model.device,
            ),
        )


    @torch.inference_mode()
    def decode_step(self, batch):
        # The token generated last step now enters the model.
        input_ids = batch.next_token_ids

        # Add one valid position to every active request.
        batch.attention_mask = torch.cat([
            batch.attention_mask,
            batch.attention_mask.new_ones(
                (len(batch.requests), 1)
            ),
        ], dim=1)

        # Each request can have a different logical position.
        position_ids = batch.sequence_lengths.unsqueeze(1)

        outputs = self.model(
            input_ids=batch.next_token_ids,
            attention_mask=batch.attention_mask,
            position_ids=position_ids,
            past_key_values=batch.past_key_values,
            cache_position=batch.cache_position,
            use_cache=True,
            logits_to_keep=1,
        )

        batch.past_key_values = outputs.past_key_values

        # The token we just fed is now part of the cache.
        batch.sequence_lengths += 1
        batch.cache_position += 1

        next_token_ids = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        batch.next_token_ids = next_token_ids

        for request, token_id in zip(batch.requests, next_token_ids):
            request.add_token(
                token_id.item(),
                self.tokenizer.eos_token_id,
            )

        return batch


    def remove_finished(self, batch):
        keep = [
            i
            for i, request in enumerate(batch.requests)
            if not request.is_finished
        ]

        if not keep:
            return None

        if len(keep) == len(batch.requests):
            return batch

        indices = torch.tensor(
            keep,
            dtype=torch.long,
            device=batch.next_token_ids.device,
        )

        batch.requests = [
            batch.requests[i]
            for i in keep
        ]

        batch.next_token_ids = batch.next_token_ids.index_select(0, indices)
        batch.attention_mask = batch.attention_mask.index_select(0, indices)
        batch.sequence_lengths = batch.sequence_lengths.index_select(0, indices)

        batch.past_key_values.batch_select_indices(indices)

        return batch