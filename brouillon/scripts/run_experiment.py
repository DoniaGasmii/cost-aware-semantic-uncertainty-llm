# scripts/run_experiment.py
"""
End-to-end runner:
- loads prompts
- runs APB generation (one-shot multi-sample branching)
- computes semantic entropy over leaf completions
- writes JSONL with candidates and metrics
"""

import argparse, json, os, pathlib, yaml
from tqdm import tqdm

from llama_backend import LlamaCppNaive
from apb_sampler import APBSampler
from se_utils import compute_se


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # --- model ---
    mcfg = cfg.get("model", {})
    model_path = str(pathlib.Path(mcfg["path"]).expanduser().resolve())
    n_ctx = int(mcfg.get("n_ctx", 2048))
    seed = int(mcfg.get("seed", 42))
    chat_template = str(mcfg.get("chat_template", "qwen"))

    backend = LlamaCppNaive(
        model_path=model_path,
        n_ctx=n_ctx,
        seed=seed,
        chat_template=chat_template,
        verbose=False,
    )

    # --- decoding ---
    dcfg = cfg.get("decoding", {})
    temperature = float(dcfg.get("temperature", 0.9))
    top_p = float(dcfg.get("top_p", 0.95))
    top_k = int(dcfg.get("top_k", 0))

    # --- apb ---
    apb = cfg.get("apb", {})
    sampler = APBSampler(
        backend=backend,
        K=int(apb.get("K", 16)),
        T=int(apb.get("T", 64)),
        B=int(apb.get("B", 48)),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        gamma=float(apb.get("gamma", 1.0)),
        alpha=float(apb.get("alpha", 0.5)),
        mass_cap=float(apb.get("mass_cap", 0.9)),
        max_children=int(apb.get("max_children", 3)),
        H_split=float(apb.get("H_split", 2.1)),
        dH_min=float(apb.get("dH_min", 0.15)),
        eos_token_id=apb.get("eos_id", None),
    )

    # --- io ---
    io = cfg.get("io", {})
    prompts_path = io["prompts_path"]
    output_path = io["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prompts = load_prompts(prompts_path)

    with open(output_path, "w", encoding="utf-8") as w:
        for item in tqdm(prompts, desc="Prompts"):
            res = sampler.generate(item["prompt"])
            se = compute_se(
                res["candidates"],
                model_name=cfg.get("semantic", {}).get("embedding_model", "all-MiniLM-L6-v2"),
                algo=cfg.get("semantic", {}).get("cluster_algorithm", "kmeans"),
                max_clusters=int(cfg.get("semantic", {}).get("max_clusters", 5)),
            )
            record = {
                "id": item["id"],
                "prompt": item["prompt"],
                "candidates": res["candidates"],
                "logprobs": res["logprobs"],
                "cluster_labels": se["labels"],
                "semantic_entropy": se["entropy"],
                "dominant_cluster_fraction": se["dominant_frac"],
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
