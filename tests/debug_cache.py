import torch
from engine.model_runner import ModelRunner


runner = ModelRunner()
model = runner.model
tokenizer = runner.tokenizer

prompt = "The capital of Canada is"
max_new_tokens = 100

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

naive_ids = inputs["input_ids"].clone()

cached_input_ids = inputs["input_ids"].clone()
cached_attention_mask = inputs["attention_mask"].clone()
past_key_values = None

cache_position = torch.arange(
    cached_input_ids.shape[1],
    dtype=torch.long,
    device=model.device,
)

for step in range(max_new_tokens):
    # Naive: recompute the entire prefix.
    with torch.no_grad():
        naive_outputs = model(
            input_ids=naive_ids,
            use_cache=False,
        )

    naive_logits = naive_outputs.logits[:, -1, :]
    naive_next = torch.argmax(naive_logits, dim=-1, keepdim=True)

    # Cached: prefill once, then only process the newest token.
    with torch.no_grad():
        cached_outputs = model(
            input_ids=cached_input_ids,
            attention_mask=cached_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )

    past_key_values = cached_outputs.past_key_values

    cached_logits = cached_outputs.logits[:, -1, :]
    cached_next = torch.argmax(cached_logits, dim=-1, keepdim=True)

    logit_diff = (
        naive_logits.float() - cached_logits.float()
    ).abs().max().item()

    print(
        f"step {step:3d} | "
        f"same token: {naive_next.item() == cached_next.item()} | "
        f"max logit diff: {logit_diff:.6f}"
    )

    if naive_next.item() != cached_next.item():
        print("\nFIRST DIVERGENCE")
        print("step:", step)

        print(
            "naive:",
            naive_next.item(),
            repr(tokenizer.decode(naive_next[0])),
        )
        print(
            "cached:",
            cached_next.item(),
            repr(tokenizer.decode(cached_next[0])),
        )

        naive_top = torch.topk(naive_logits[0], k=5)
        cached_top = torch.topk(cached_logits[0], k=5)

        print("\nNaive top 5:")
        for token_id, logit in zip(naive_top.indices, naive_top.values):
            print(
                repr(tokenizer.decode(token_id)),
                float(logit),
            )

        print("\nCached top 5:")
        for token_id, logit in zip(cached_top.indices, cached_top.values):
            print(
                repr(tokenizer.decode(token_id)),
                float(logit),
            )

        break

    # Both picked the same token, so the prefixes remain identical.
    naive_ids = torch.cat([naive_ids, naive_next], dim=1)

    cached_input_ids = cached_next

    cached_attention_mask = torch.cat([
        cached_attention_mask,
        cached_attention_mask.new_ones((1, 1)),
    ], dim=1)

    cache_position = cache_position[-1:] + 1
else:
    print("\nNo divergence found.")