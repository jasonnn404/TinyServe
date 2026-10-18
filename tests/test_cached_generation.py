from engine.model_runner import ModelRunner

runner = ModelRunner()

prompt = "The capital of Canada is"

print("Naive:")
print(runner.generate_naive(prompt, max_new_tokens=20))

print("\nKV cache:")
print(runner.generate_cached(prompt, max_new_tokens=20))