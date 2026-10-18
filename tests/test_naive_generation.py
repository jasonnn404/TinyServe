from engine.model_runner import ModelRunner

runner = ModelRunner()

prompt = "The capital of Canada is"
text = runner.generate_naive(prompt, max_new_tokens=20)

print(text)