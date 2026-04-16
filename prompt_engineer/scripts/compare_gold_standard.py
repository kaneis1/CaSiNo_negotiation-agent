import json
from prompt_engineer.llm.client import LlamaClient
from prompt_engineer.core.classify_strategy import benchmark
from prompt_engineer.core.classify_strategy import build_system_prompt
# Load annotated data
with open('CaSiNo/data/casino_ann.json') as f:
    dialogues = json.load(f)

# Load model from scratch
client = LlamaClient(
    model_id='/sc/arion/scratch/cuiz02/hf_cache/transformers/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b',
    temperature=0.0,
    max_new_tokens=128,
    system_prompt=build_system_prompt()
)

# Run benchmark (start with 5 dialogues to test first)
results = benchmark(dialogues, client)
print(results)
