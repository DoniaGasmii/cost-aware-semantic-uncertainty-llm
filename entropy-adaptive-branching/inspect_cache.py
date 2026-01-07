"""Quick script to inspect DynamicCache structure."""

import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate cache
input_ids = tokenizer.encode("Hello world", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    cache = outputs.past_key_values

print(f"Cache type: {type(cache)}")
print(f"Cache attributes: {dir(cache)}")
print(f"\nKey attributes:")
for attr in dir(cache):
    if not attr.startswith('_'):
        print(f"  {attr}: {type(getattr(cache, attr, None))}")

# Check if it has data
if hasattr(cache, '__len__'):
    print(f"\nCache length: {len(cache)}")

# Try to access data
try:
    print(f"\nTrying cache[0]: {type(cache[0])}")
    if cache[0] is not None:
        print(f"cache[0] shape: {cache[0][0].shape if isinstance(cache[0], tuple) else 'not a tuple'}")
except Exception as e:
    print(f"Error accessing cache[0]: {e}")

# Check get_seq_length
try:
    seq_len = cache.get_seq_length()
    print(f"\nget_seq_length(): {seq_len}")
except Exception as e:
    print(f"Error calling get_seq_length(): {e}")

# Check to_legacy_cache
try:
    legacy = cache.to_legacy_cache()
    print(f"\nto_legacy_cache(): {type(legacy)}")
    if legacy:
        print(f"Legacy cache length: {len(legacy)}")
        print(f"Legacy first layer shape: {legacy[0][0].shape}")
except Exception as e:
    print(f"Error calling to_legacy_cache(): {e}")
