"""
demo.py — Visual demo for Adaptive Probabilistic Branching (APB)
Shows how entropy drives branching and prints final sampled completions.
"""

import sys
import numpy as np
from scripts.llama_backend import LlamaCppNaive
from scripts.apb_sampler import APBSampler, entropy_from_probs, nucleus_filter


def visualize_branching(prompt: str, model_path: str = "models/qwen2.5-7b-instruct-q4_k_m.gguf"):
    print("=" * 80)
    print(f"Prompt: {prompt!r}")
    print("=" * 80)

    # ---- setup ----
    backend = LlamaCppNaive(model_path=model_path, n_ctx=4096, verbose=False, chat_template="qwen")
    sampler = APBSampler(backend=backend, K=5, T=32, H_split=0.4, dH_min=0.15, max_children=3)

    # initialize
    sid = backend.init_sequence(prompt)
    base_len = backend.length(sid)
    active = [sid]
    step = 0
    print("\n[ Step-by-step decoding ]\n")

    # keep all sequences for final completions
    all_nodes = {sid: {"text": prompt, "parent": None}}

    while active:
        step += 1
        logits_batch = backend.forward_last(active)
        next_active = []

        for seq_id, logits in zip(active, logits_batch):
            z = np.asarray(logits, dtype=np.float32)
            z = z - float(z.max())
            probs = np.exp(z) / (np.exp(z).sum() + 1e-12)

            cand = nucleus_filter(probs, top_p=0.9)
            p_cand = probs[cand]
            H = entropy_from_probs(p_cand)
            Hmax = np.log(len(cand) or 1)

            # pick according to entropy
            if H > sampler.H_split:
                branch_type = "🌿 BRANCH"
                picks = cand[:min(3, len(cand))]
            else:
                branch_type = "→ single"
                picks = cand[:1]

            tok_strs = [backend._detok([int(t)]).strip() or "<space>" for t in picks]

            print(f"Step {step:02d} | Entropy={H:.2f} | {branch_type} | Tokens={tok_strs}")

            # fork children
            for tok in picks:
                child = backend.fork_and_append(seq_id, int(tok))
                all_nodes[child] = {"text": backend.text_of(child), "parent": seq_id}
                if backend.length(child) < base_len + sampler.T:
                    next_active.append(child)

        if not next_active or step > sampler.T:
            break
        active = next_active

    print("\n" + "=" * 80)
    print("Finished decoding.\n")

    # ---- Final completions ----
    completions = []
    for sid, info in all_nodes.items():
        if backend.length(sid) >= base_len + sampler.T:
            text = backend.text_of(sid)
            completions.append(text)

    if not completions:
        completions = [backend.text_of(sid) for sid in active or all_nodes.keys()]

    print("[ Final sampled responses ]")
    for i, txt in enumerate(completions[:sampler.K], 1):
        print(f"\n--- Candidate #{i} ---")
        print(backend.answer_only(txt).strip())

    print("\n" + "=" * 80)


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Overfitting is"
    visualize_branching(prompt)
