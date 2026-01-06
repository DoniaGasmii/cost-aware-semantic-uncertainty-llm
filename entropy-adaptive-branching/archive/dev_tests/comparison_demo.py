#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eab import EntropyAdaptiveBranching

MODEL_NAME = "Qwen/Qwen3-8B"
PROMPT = "what is the capital of Tunisia?"
MAX_PATHS = 25
MAX_NEW_TOKENS = 30

def get_memory_usage():
    return psutil.Process().memory_info().rss / 1024 / 1024

def naive_generation(model, tokenizer, prompt, num_samples, max_new_tokens, temperature=0.8, seed=42):
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = [
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            ) for _ in range(num_samples)
        ]
    return [tokenizer.decode(o[0], skip_special_tokens=True) for o in outputs]

# --- EAB ---
mem_before = get_memory_usage()
start = time.time()
eab = EntropyAdaptiveBranching(
    model_name=MODEL_NAME,
    device="cpu",
    entropy_threshold=0.1,
    branch_factor=3,
    max_paths=MAX_PATHS
)
eab_results = eab.generate(prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, seed=42, show_progress=False)
eab_time = time.time() - start
eab_mem = get_memory_usage() - mem_before
del eab
torch.cuda.empty_cache()

# --- Naive ---
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_samples = len(eab_results)
mem_before = get_memory_usage()
start = time.time()
naive_results = naive_generation(model, tokenizer, PROMPT, num_samples, MAX_NEW_TOKENS)
naive_time = time.time() - start
naive_mem = get_memory_usage() - mem_before
del model, tokenizer
torch.cuda.empty_cache()

# --- Table ---
speedup = naive_time / eab_time if eab_time > 0 else 0
mem_reduction = ((naive_mem - eab_mem) / naive_mem * 100) if naive_mem > 0 else 0

print(f"{'Metric':<30} {'EAB':<15} {'Naive':<15} {'Speedup':<15}")
print("-" * 80)
print(f"{'Time (seconds)':<30} {eab_time:<15.2f} {naive_time:<15.2f} {speedup:.2f}x faster")
print(f"{'Memory (MB)':<30} {eab_mem:<15.1f} {naive_mem:<15.1f} {mem_reduction:.1f}% less")
print(f"{'Samples generated':<30} {len(eab_results):<15} {len(naive_results):<15} {'SAME'}")