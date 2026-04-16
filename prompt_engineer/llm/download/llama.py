import transformers
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Say hello in one sentence."},
]

outputs = pipe(messages, max_new_tokens=128)
print(outputs[0]["generated_text"][-1]["content"])
