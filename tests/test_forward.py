import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

prompt = "The capital of Canada is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)

print("input_ids:", inputs["input_ids"])
print("input shape:", inputs["input_ids"].shape)
print("logits shape:", logits.shape)
print("next token id:", next_token_id.item())
print("next token:", repr(tokenizer.decode(next_token_id)))