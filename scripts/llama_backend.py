# scripts/llama_backend.py
"""
Minimal llama.cpp backend with a forkable API surface.
- Supports a Qwen-style chat template or plain prompts.
- Re-evaluates prefix per node to obtain logits for the next token (Python binding friendly).
- Designed to be swapped with a true multi-sequence KV backend later without changing APB.
"""

from typing import List, Dict, Any, Optional, Protocol
import numpy as np
    
from collections import deque

import sys
sys.path.append("llama-cpp-python")  # for local imports in demo environment
try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None


# ------------------ prompt templating ------------------

def apply_chat_template(user_text: str, template: str) -> str:
    if template == "qwen":
        return (
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    # "plain": leave as-is
    return user_text


def strip_to_assistant(text: str, template: str) -> str:
    if template == "qwen":
        marker = "<|im_start|>assistant\n"
        i = text.rfind(marker)
        return text[i + len(marker):] if i != -1 else text
    return text


# ------------------ backend protocol ------------------

class ForkableBackend(Protocol):
    def init_sequence(self, prompt: str) -> int: ...
    def forward_last(self, seq_ids: List[int]) -> List[np.ndarray]: ...
    def fork_and_append(self, parent_seq_id: int, token_id: int) -> int: ...
    def length(self, seq_id: int) -> int: ...
    def text_of(self, seq_id: int) -> str: ...
    def last_token_was_eos(self, seq_id: int, eos_id: int) -> bool: ...
    def answer_only(self, text: str) -> str: ...


# ------------------ naive llama.cpp backend ------------------

class LlamaCppNaive:
    """
    Re-evaluates the full prefix for each node to get next-token logits.
    This is compute-heavier than true KV sharing but is simple and correct.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        seed: int = 42,
        chat_template: str = "qwen",  # "qwen" | "plain"
        verbose: bool = False,
    ):
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            logits_all=True,
            seed=seed,
            verbose=verbose,
        )
        self.chat_template = chat_template
        self._store: Dict[int, Dict[str, Any]] = {}  # seq_id -> {"text": str, "token_ids": List[int]}
        self._seq_next = 1

    # ---- helpers ----
    def _tokenize(self, s: str) -> List[int]:
        return self.llm.tokenize(s.encode("utf-8"), add_bos=False)

    def _detok(self, ids: List[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="ignore")

    # ---- protocol ----
    def init_sequence(self, user_prompt: str) -> int:
        chat = apply_chat_template(user_prompt, self.chat_template)
        sid = self._seq_next; self._seq_next += 1
        toks = self._tokenize(chat)
        self._store[sid] = {"text": chat, "token_ids": toks}
        return sid

    def forward_last(self, seq_ids: list[int]) -> list[np.ndarray]:
        outs: list[np.ndarray] = []
        for sid in seq_ids:
            entry = self._store[sid]
            toks = entry["token_ids"]

            # Reset model state
            if hasattr(self.llm, "reset"):
                self.llm.reset()

            # Evaluate: this fills self.llm.eval_logits (a deque)
            self.llm.eval(toks)

            logits = None

            # 1️⃣ current public API (v0.3.16) → eval_logits is a deque
            if hasattr(self.llm, "eval_logits") and isinstance(self.llm.eval_logits, deque):
                if len(self.llm.eval_logits) > 0:
                    logits = np.array(self.llm.eval_logits[-1], dtype=np.float32)

            # 2️⃣ alternate getter (some 0.3.x Linux builds)
            elif hasattr(self.llm, "get_logits"):
                logits = np.array(self.llm.get_logits(), dtype=np.float32)

            # 3️⃣ legacy attribute (old <0.2.70)
            elif hasattr(self.llm, "logits") and self.llm.logits is not None:
                logits = np.array(self.llm.logits[-1], dtype=np.float32)

            if logits is None:
                raise RuntimeError("Cannot access logits — unsupported llama-cpp-python build")

            # take last-token slice only
            n_vocab_attr = getattr(self.llm, "n_vocab", None)
            if callable(n_vocab_attr):
                n_vocab = n_vocab_attr()
            else:
                n_vocab = n_vocab_attr

            if isinstance(n_vocab, int) and logits.size >= n_vocab:
                logits = logits[-n_vocab:]

            outs.append(logits)

        return outs



    def fork_and_append(self, parent_seq_id: int, token_id: int) -> int:
        p = self._store[parent_seq_id]
        new_ids = p["token_ids"] + [int(token_id)]
        new_text = p["text"] + self._detok([int(token_id)])
        sid = self._seq_next; self._seq_next += 1
        self._store[sid] = {"text": new_text, "token_ids": new_ids}
        return sid

    def length(self, seq_id: int) -> int:
        return len(self._store[seq_id]["token_ids"])

    def text_of(self, seq_id: int) -> str:
        return self._store[seq_id]["text"]

    def last_token_was_eos(self, seq_id: int, eos_id: int) -> bool:
        ids = self._store[seq_id]["token_ids"]
        return bool(ids and (ids[-1] == int(eos_id)))

    def answer_only(self, text: str) -> str:
        return strip_to_assistant(text, self.chat_template)
