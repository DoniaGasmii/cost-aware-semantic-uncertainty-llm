# scripts/run_comparison.py
"""
Compare:
  (A) Naive sampling: 10 independent completions (re-process prompt each time)
  (B) APB + shared prefix: adaptive branching with KV reuse simulation

Logs:
  - Semantic entropy
  - Wall time, CPU time, peak memory (approx)
"""

import argparse
import json
import os
import time
import psutil
import yaml
from tqdm import tqdm

from llama_backend import LlamaCppNaive
from apb_sampler import APBSampler
from se_utils import compute_se


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def naive_sample(backend, prompt: str, n: int, **decoding_kwargs):
    candidates, logprobs = [], []
    for _ in range(n):
        out = backend.llm(
            prompt,
            max_tokens=64,
            temperature=decoding_kwargs.get("temperature", 0.9),
            top_p=decoding_kwargs.get("top_p", 0.95),
            top_k=decoding_kwargs.get("top_k", 0),
            repeat_penalty=1.1,
            stop=["\n", "</s>", "User:", "Assistant:"],
        )
        text = out["choices"][0]["text"].strip()
        candidates.append(text)
        logprobs.append(0.0)  # not tracked in naive
    return {"candidates": candidates, "logprobs": logprobs}


def measure_resources(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2  # MB
    cpu_start = time.process_time()
    wall_start = time.time()

    result = func(*args, **kwargs)

    wall_time = time.time() - wall_start
    cpu_time = time.process_time() - cpu_start
    mem_after = process.memory_info().rss / 1024**2  # MB
    peak_mem = max(mem_after, mem_before)

    return result, {
        "wall_time_sec": round(wall_time, 3),
        "cpu_time_sec": round(cpu_time, 3),
        "peak_memory_mb": round(peak_mem, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- model ---
    mcfg = cfg["model"]
    model_path = str(os.path.expanduser(mcfg["path"]))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    backend = LlamaCppNaive(
        model_path=model_path,
        n_ctx=int(mcfg.get("n_ctx", 2048)),
        seed=int(mcfg.get("seed", 42)),
        chat_template=mcfg.get("chat_template", "qwen"),
        verbose=False,
    )

    # --- decoding ---
    dcfg = cfg["decoding"]
    decoding_kwargs = {
        "temperature": float(dcfg.get("temperature", 0.9)),
        "top_p": float(dcfg.get("top_p", 0.95)),
        "top_k": int(dcfg.get("top_k", 0)),
    }

    # --- APB ---
    apb_cfg = cfg["apb"]
    apb_sampler = APBSampler(
        backend=backend,
        K=int(apb_cfg.get("K", 10)),
        T=int(apb_cfg.get("T", 64)),
        B=int(apb_cfg.get("B", 32)),
        temperature=decoding_kwargs["temperature"],
        top_p=decoding_kwargs["top_p"],
        top_k=decoding_kwargs["top_k"],
        gamma=float(apb_cfg.get("gamma", 1.0)),
        alpha=float(apb_cfg.get("alpha", 0.5)),
        mass_cap=float(apb_cfg.get("mass_cap", 0.9)),
        max_children=int(apb_cfg.get("max_children", 3)),
        H_split=float(apb_cfg.get("H_split", 2.1)),
        dH_min=float(apb_cfg.get("dH_min", 0.15)),
        eos_token_id=apb_cfg.get("eos_id", 151645),  # Qwen EOS
        rng=None,
    )

    # --- semantic ---
    sem_cfg = cfg.get("semantic", {})
    embed_model = sem_cfg.get("embedding_model", "all-MiniLM-L6-v2")
    cluster_algo = sem_cfg.get("cluster_algorithm", "kmeans")
    max_clusters = int(sem_cfg.get("max_clusters", 5))

    # --- I/O ---
    io_cfg = cfg["io"]
    prompts = load_prompts(io_cfg["prompts_path"])
    output_dir = os.path.dirname(io_cfg["output_path"])
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for item in tqdm(prompts, desc="Prompts"):
        prompt = item["prompt"]

        # --- Naive: 10 independent samples ---
        naive_res, naive_time = measure_resources(
            naive_sample, backend, prompt, 10, **decoding_kwargs
        )
        naive_se = compute_se(
            naive_res["candidates"], embed_model, cluster_algo, max_clusters
        )

        # --- APB: shared prefix tree ---
        apb_res, apb_time = measure_resources(
            apb_sampler.generate, prompt
        )
        apb_se = compute_se(
            apb_res["candidates"], embed_model, cluster_algo, max_clusters
        )

        record = {
            "id": item["id"],
            "prompt": prompt,
            "naive": {
                "candidates": naive_res["candidates"],
                "semantic_entropy": naive_se["entropy"],
                "dominant_cluster_fraction": naive_se["dominant_frac"],
                "timing": naive_time,
            },
            "apb": {
                "candidates": apb_res["candidates"],
                "semantic_entropy": apb_se["entropy"],
                "dominant_cluster_fraction": apb_se["dominant_frac"],
                "timing": apb_time,
            },
            "speedup_wall": round(naive_time["wall_time_sec"] / apb_time["wall_time_sec"], 2),
        }
        results.append(record)

        # Print live summary
        print("\n" + "="*60)
        print(f"Prompt: {item['id']}")
        print(f"Naive SE: {naive_se['entropy']:.3f} | Time: {naive_time['wall_time_sec']}s")
        print(f"APB   SE: {apb_se['entropy']:.3f} | Time: {apb_time['wall_time_sec']}s")
        print(f"Speedup: {record['speedup_wall']}x")

    # Save full results
    with open(io_cfg["output_path"], "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Results saved to {io_cfg['output_path']}")


if __name__ == "__main__":
    main()