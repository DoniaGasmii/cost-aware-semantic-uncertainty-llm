"""
Quick comparison of naive multi-sampling vs Entropy-Adaptive Branching (EAB)
on vLLM with prefix caching enabled.

This script is meant as a lightweight benchmark harness; it reuses one LLM
instance to avoid reloading weights. RSS memory is sampled via psutil if
installed; otherwise memory is reported as None.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from vllm import LLM, SamplingParams

from entropy_adaptive_branching import EntropyAdaptiveBrancher


def _rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def naive_generate(
    llm: LLM,
    prompt: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        logprobs=5,
    )
    prompts = [prompt] * n_samples
    t0 = time.time()
    mem0 = _rss_mb()
    outputs = llm.generate(prompts, params)
    mem1 = _rss_mb()
    duration = time.time() - t0

    completions: List[str] = []
    for out in outputs:
        text = prompt + out.outputs[0].text
        completions.append(text)

    return {
        "duration_sec": duration,
        "memory_mb": mem1,
        "completions": completions,
        "samples": len(completions),
    }


def run_comparison() -> None:
    prompt = "Question: What is the capital of France? Answer:"
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    temperature = 0.8
    top_p = 0.95
    max_new_tokens = 48
    n_samples = 20

    print("Loading vLLM model once...")
    llm = LLM(
        model=model_name,
        enable_prefix_caching=True,
    )

    print("\nRunning naive sampling...")
    naive_stats = naive_generate(
        llm=llm,
        prompt=prompt,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"Naive: {naive_stats['samples']} samples in {naive_stats['duration_sec']:.2f}s, "
          f"RSS={naive_stats['memory_mb'] or 'n/a'} MB")

    print("\nRunning entropy-adaptive branching...")
    brancher = EntropyAdaptiveBrancher(
        model_name=model_name,
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=20,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    t0 = time.time()
    mem0 = _rss_mb()
    eab_stats = brancher.generate(prompt=prompt, target_samples=n_samples)
    mem1 = _rss_mb()
    duration = time.time() - t0

    print(f"EAB: {len(eab_stats['completions'])} samples in {duration:.2f}s, "
          f"RSS={mem1 or 'n/a'} MB, steps={eab_stats['steps']}")

    print("\nSample outputs (first 3 naive vs first 3 EAB):")
    for i in range(3):
        naive_text = naive_stats['completions'][i][len(prompt):].strip()
        eab_text = eab_stats['completions'][i][len(prompt):].strip()
        print(f"Naive {i+1}: {naive_text}")
        print(f"EAB   {i+1}: {eab_text}")
        print("-" * 40)


if __name__ == "__main__":
    run_comparison()
