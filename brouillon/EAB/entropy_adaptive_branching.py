"""
Entropy-Adaptive Branching (EAB) generation using vLLM prefix caching.

Key idea:
- Encode the prompt once, let vLLM share prefix KV blocks.
- Autoregressively expand paths; when next-token entropy is high, branch into
  multiple continuations while reusing the shared prefix cache.
- Track per-path logprobs; prune to `max_paths` by highest logprob.

This focuses on correctness and clarity over extreme micro-optimizations; vLLM's
prefix caching already deduplicates prompt/prefix KV blocks across requests.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import Logprob


@dataclass
class GenerationPath:
    tokens: List[int] = field(default_factory=list)
    logprob: float = 0.0
    finished: bool = False


class EntropyAdaptiveBrancher:
    """
    Entropy-triggered branching generator using vLLM.

    Branching policy:
    - Compute next-token entropy from top-k logprobs (approximate).
    - If normalized entropy >= entropy_threshold and under max_paths, fork
      `branch_factor` child paths on the top tokens.
    - Otherwise continue along the best token.
    - Paths stop on EOS or when max_new_tokens is reached.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size: int = 1,
        entropy_threshold: float = 0.4,
        branch_factor: int = 3,
        max_paths: int = 20,
        max_new_tokens: int = 64,
        top_logprobs: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.model_name = model_name
        self.entropy_threshold = entropy_threshold
        self.branch_factor = branch_factor
        self.max_paths = max_paths
        self.max_new_tokens = max_new_tokens
        self.top_logprobs = top_logprobs
        self.temperature = temperature
        self.top_p = top_p

        # vLLM shares prompt/prefix KV automatically when enable_prefix_caching is True.
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        # Use EOS from tokenizer; fall back to None.
        self.eos_token_ids = set(
            [tid for tid in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] if tid is not None]
        )

    @staticmethod
    def _normalized_entropy_from_logprobs(logprob_dict: Dict[int, Logprob], vocab_size: int) -> float:
        """
        Approximate normalized entropy using the provided top-k logprobs.
        If top-k << vocab_size, this underestimates entropy slightly.
        """
        if not logprob_dict:
            return 0.0
        logps = np.array([lp.logprob for lp in logprob_dict.values()])
        lse = np.log(np.sum(np.exp(logps)))
        probs = np.exp(logps - lse)
        entropy = -float(np.sum(probs * (logps - lse)))  # H(p) over observed mass
        # Normalize by log(V); clip to [0, 1] for stability.
        norm = math.log(max(vocab_size, 2))
        return max(0.0, min(1.0, entropy / norm))

    def _decode(self, prompt_ids: List[int], path: GenerationPath) -> str:
        full = prompt_ids + path.tokens
        return self.tokenizer.decode(full, skip_special_tokens=True)

    def _sample_next(
        self,
        prompt_ids: List[int],
        path: GenerationPath,
    ) -> Tuple[int, float, Dict[int, Logprob]]:
        """
        Generate the next token for a single path using vLLM with max_tokens=1.
        Returns token_id, token_logprob, logprob_dict_for_entropy.
        """
        params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1,
            logprobs=self.top_logprobs,
        )

        outputs = self.llm.generate(
            prompt_token_ids=[prompt_ids + path.tokens],
            sampling_params=params,
        )
        out = outputs[0].outputs[0]
        token_id = out.token_ids[0]
        # vLLM returns a list with one dict (per generated token)
        token_logprob_dict = out.logprobs[0]
        token_logprob = token_logprob_dict[token_id].logprob
        return token_id, token_logprob, token_logprob_dict

    def generate(
        self,
        prompt: str,
        target_samples: int = 20,
    ) -> Dict[str, object]:
        """
        Run entropy-adaptive branching until we hit max_new_tokens or all paths finish.

        Returns:
            {
                "completions": List[str],
                "paths": List[GenerationPath],
                "steps": int,
                "duration_sec": float,
            }
        """
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        vocab_size = len(self.tokenizer)
        paths: List[GenerationPath] = [GenerationPath()]

        start_time = time.time()
        steps = 0

        for _ in range(self.max_new_tokens):
            steps += 1
            new_paths: List[GenerationPath] = []

            for path in paths:
                if path.finished:
                    new_paths.append(path)
                    continue

                token_id, token_logprob, logprob_dict = self._sample_next(prompt_ids, path)
                entropy = self._normalized_entropy_from_logprobs(logprob_dict, vocab_size)

                # Stop if EOS/pad reached
                if token_id in self.eos_token_ids:
                    path.finished = True
                    new_paths.append(path)
                    continue

                should_branch = (
                    entropy >= self.entropy_threshold
                    and len(paths) < self.max_paths
                )

                if should_branch:
                    # Select top tokens to branch on
                    sorted_tokens = sorted(
                        logprob_dict.items(),
                        key=lambda kv: kv[1].logprob,
                        reverse=True,
                    )[: self.branch_factor]

                    for idx, (tid, lp_obj) in enumerate(sorted_tokens):
                        child_tokens = path.tokens + [tid]
                        child_logprob = path.logprob + lp_obj.logprob
                        child_path = GenerationPath(tokens=child_tokens, logprob=child_logprob)
                        new_paths.append(child_path)
                else:
                    # Continue with best token
                    best_tid, best_lp_obj = max(
                        logprob_dict.items(), key=lambda kv: kv[1].logprob
                    )
                    path.tokens.append(best_tid)
                    path.logprob += best_lp_obj.logprob
                    new_paths.append(path)

            # Prune to top-k paths if we exceed max_paths
            if len(new_paths) > self.max_paths:
                new_paths = sorted(new_paths, key=lambda p: p.logprob, reverse=True)[: self.max_paths]

            paths = new_paths

            # Early exit if we already have enough finished paths
            finished = [p for p in paths if p.finished]
            if len(finished) >= target_samples:
                break

        duration_sec = time.time() - start_time

        # If we didn't finish enough paths, take the top ones.
        sorted_paths = sorted(paths, key=lambda p: p.logprob, reverse=True)
        completions = [self._decode(prompt_ids, p) for p in sorted_paths[:target_samples]]

        return {
            "completions": completions,
            "paths": sorted_paths[:target_samples],
            "steps": steps,
            "duration_sec": duration_sec,
        }


def main_demo() -> None:
    prompt = "Question: What is the capital of France? Answer:"
    brancher = EntropyAdaptiveBrancher(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        entropy_threshold=0.4,
        branch_factor=3,
        max_paths=20,
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.95,
    )
    result = brancher.generate(prompt, target_samples=10)
    print(f"EAB finished in {result['duration_sec']:.2f}s, steps={result['steps']}")
    for i, comp in enumerate(result["completions"]):
        print(f"{i+1:02d}: {comp[len(prompt):].strip()}")


if __name__ == "__main__":
    main_demo()
