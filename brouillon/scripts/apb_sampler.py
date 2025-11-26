# scripts/apb_sampler.py
"""
Adaptive Probabilistic Branching (APB) sampler.
- Grows a small decoding tree in one pass.
- Branches only when next-token entropy suggests uncertainty.
- Leaves serve as multiple candidate completions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import numpy as np


# ---------------- utils ----------------

def entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def nucleus_filter(probs: np.ndarray, top_p: float = 0.95, top_k: int = 0) -> np.ndarray:
    order = np.argsort(-probs)
    if top_k and top_k > 0:
        order = order[:top_k]
    cum = 0.0
    kept = []
    for idx in order:
        kept.append(idx)
        cum += float(probs[idx])
        if cum >= top_p:
            break
    if not kept:
        kept = [int(np.argmax(probs))]
    return np.array(kept, dtype=np.int32)

def sample_without_replacement(indices: np.ndarray, probs: np.ndarray, b: int, gamma: float, rng: np.random.Generator) -> List[int]:
    # weights ~ p^gamma, normalized
    w = np.power(probs[indices].clip(1e-12), max(gamma, 1e-6))
    w = w / (w.sum() + 1e-12)
    picks: List[int] = []
    pool = indices.copy()
    weights = w.copy()
    for _ in range(min(b, len(pool))):
        j = int(rng.choice(len(pool), p=weights))
        picks.append(int(pool[j]))
        pool = np.delete(pool, j)
        weights = np.delete(weights, j)
        s = float(weights.sum())
        if s > 0:
            weights = weights / s
        else:
            break
    return picks


# ---------------- data ----------------

@dataclass
class Node:
    seq_id: int
    logprob: float = 0.0
    done: bool = False
    entropy_prev: Optional[float] = None
    meta: dict = field(default_factory=dict)


# ---------------- APB ----------------

class APBSampler:
    def __init__(
        self,
        backend,                         # ForkableBackend
        K: int = 10,                     # target number of leaves
        T: int = 32,                     # per-leaf token budget
        B: int = 48,                     # cap of active nodes
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: int = 0,
        gamma: float = 1.0,              # diversity exponent (1.0 = off)
        alpha: float = 0.5,              # mass growth with entropy
        mass_cap: float = 0.9,           # never exceed this mass
        max_children: int = 3,           # cap children per node
        H_split: float = 2.1,            # entropy threshold to branch
        dH_min: float = 0.15,            # require min increase vs prev
        eos_token_id: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.backend = backend
        self.K, self.T, self.B = int(K), int(T), int(B)
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.gamma, self.alpha = float(gamma), float(alpha)
        self.mass_cap, self.max_children = float(mass_cap), int(max_children)
        self.H_split, self.dH_min = float(H_split), float(dH_min)
        self.eos = eos_token_id
        self.rng = rng or np.random.default_rng()

    # mass target grows with entropy (normalized by log(|cand|))
    def _mass_target(self, H: float, Hmax: float) -> float:
        if Hmax <= 1e-9:
            return 0.5
        frac = max(0.0, min(1.0, H / Hmax))
        return min(self.mass_cap, max(0.5, 0.5 + self.alpha * frac))

    # branching decision: need both absolute entropy & delta increase
    def _should_branch(self, H_now: float, H_prev: Optional[float]) -> bool:
        if H_now < self.H_split:
            return False
        if H_prev is None:
            return True
        return (H_now - H_prev) >= self.dH_min

    def generate(self, prompt: str) -> Dict[str, Any]:
        # small bias: if prompt ends with "A:", add a space so first token is likely natural text
        if prompt.rstrip().endswith("A:"):
            prompt = prompt + " "

        # initialize root
        root_id = self.backend.init_sequence(prompt)
        base_len = self.backend.length(root_id)

        active: List[Node] = [Node(seq_id=root_id)]
        completed: List[Node] = []

        while len(completed) < self.K and active:
            seq_batch = [n.seq_id for n in active]
            logits_batch = self.backend.forward_last(seq_batch)

            next_active: List[Node] = []

            for node, logits in zip(active, logits_batch):
                if node.done:
                    completed.append(node)
                    continue

                # temperature softmax (stable)
                z = np.asarray(logits, dtype=np.float32).ravel()
                z = z - float(z.max())
                exp = np.exp(z / max(self.temperature, 1e-6), dtype=np.float64)
                probs = (exp / (exp.sum() + 1e-12)).astype(np.float32)

                # candidate set
                cand = nucleus_filter(probs, self.top_p, self.top_k)
                p_cand = probs[cand]
                H = entropy_from_probs(p_cand)
                Hmax = math.log(max(1, cand.size))

                # stop conditions (token budget or EOS)
                if (self.eos is not None and self.backend.last_token_was_eos(node.seq_id, self.eos)) or \
                   (self.backend.length(node.seq_id) >= base_len + self.T):
                    node.done = True
                    completed.append(node)
                    continue

                # build a small cover to target mass
                target_mass = self._mass_target(H, Hmax)
                order = cand[np.argsort(-p_cand)]  # descending by prob
                cover, cum = [], 0.0
                for t in order:
                    t = int(t)
                    cover.append(t)
                    cum += float(probs[t])
                    if cum >= target_mass or len(cover) >= self.max_children:
                        break

                # decide branching or single continuation
                if self._should_branch(H, node.entropy_prev):
                    # diversity sampling without replacement over "cover"
                    picks = sample_without_replacement(
                        np.array(cover, dtype=np.int32),
                        probs,
                        b=len(cover),
                        gamma=self.gamma,
                        rng=self.rng,
                    )
                else:
                    picks = [int(cover[0])]  # greedy

                # create children
                for tok in picks:
                    child_id = self.backend.fork_and_append(node.seq_id, int(tok))
                    done_flag = (
                        (self.eos is not None and self.backend.last_token_was_eos(child_id, self.eos)) or
                        (self.backend.length(child_id) >= base_len + self.T)
                    )
                    child = Node(
                        seq_id=child_id,
                        logprob=node.logprob + math.log(max(float(probs[int(tok)]), 1e-12)),
                        done=done_flag,
                        entropy_prev=H,
                    )
                    if child.done:
                        completed.append(child)
                    else:
                        next_active.append(child)

            # keep only top-B active
            if len(next_active) > self.B:
                next_active.sort(key=lambda n: n.logprob, reverse=True)
                next_active = next_active[: self.B]

            active = next_active

            # safety—if nothing left but no completed, stop
            if not active and len(completed) == 0:
                break

        # top-up with partials if we didn't reach K
        if len(completed) < self.K and active:
            active.sort(key=lambda n: n.logprob, reverse=True)
            completed.extend(active[: self.K - len(completed)])

        # render texts (assistant-only if chat template)
        texts = [self.backend.text_of(n.seq_id) for n in completed[: self.K]]

        def extract_after_A(assistant_text: str) -> str:
            # backend.answer_only strips system/user if needed
            a = self.backend.answer_only(assistant_text)
            i = a.rfind("A:")
            return a[i + 2:].strip() if i != -1 else a.strip()

        completions = [extract_after_A(t) for t in texts]
        return {"candidates": completions, "logprobs": [n.logprob for n in completed[: self.K]]}
