"""
Debug script to trace COW cache behavior and identify the issue.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eab.core_cow import EntropyAdaptiveBranching
from eab.cache_cow import CopyOnWriteCache


def debug_cow_generation():
    """Run a simple generation with debug output."""

    print("="*80)
    print("DEBUG: COW Cache Generation")
    print("="*80)

    # Simple test
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    prompt = "The capital of France is"

    eab = EntropyAdaptiveBranching(
        model_name,
        entropy_threshold=0.4,  # High threshold to minimize branching
        branch_factor=2,
        max_paths=5,
        torch_dtype=torch.float16
    )

    # Encode prompt
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = eab.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = eab.tokenizer.encode(formatted_prompt, return_tensors="pt").to(eab.device)

    print(f"\nPrompt: '{prompt}'")
    print(f"Formatted length: {input_ids.shape[1]} tokens")

    # Initial forward pass
    with torch.no_grad():
        outputs = eab.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    print(f"\nInitial logits shape: {logits.shape}")
    print(f"Initial logits top-5 values: {torch.topk(logits[0], 5).values}")
    print(f"Initial past_key_values type: {type(past_key_values)}")

    # Wrap in COW cache
    if isinstance(past_key_values, tuple):
        cow_cache = CopyOnWriteCache.from_legacy_cache(past_key_values, device=eab.device)
    else:
        cow_cache = CopyOnWriteCache.from_legacy_cache(
            past_key_values.to_legacy_cache(),
            device=eab.device
        )

    print(f"\nCOW cache created:")
    print(f"  Sequence length: {cow_cache.get_seq_length()}")
    print(f"  Own cache layers: {len(cow_cache.own_cache.key_cache) if hasattr(cow_cache.own_cache, 'key_cache') else 0}")

    # Test token generation
    print(f"\n{'='*80}")
    print("Testing single token generation")
    print(f"{'='*80}")

    # Sample a token
    probs = torch.nn.functional.softmax(logits[0], dim=-1)
    token_id = torch.multinomial(probs, 1).item()
    token_text = eab.tokenizer.decode([token_id])

    print(f"\nGenerated token: {token_id} ('{token_text}')")
    print(f"Token probability: {probs[token_id].item():.4f}")

    # Now generate next token using the cache
    print(f"\n{'='*80}")
    print("Generating second token with cache")
    print(f"{'='*80}")

    # Convert COW cache to model format
    legacy_cache = cow_cache.to_legacy_cache()
    print(f"\nConverted to legacy cache: {len(legacy_cache)} layers")
    print(f"First layer key shape: {legacy_cache[0][0].shape}")

    from transformers import DynamicCache
    cache_for_model = DynamicCache.from_legacy_cache(legacy_cache)

    # Pass token to model with cache
    token_tensor = torch.tensor([[token_id]]).to(eab.device)

    with torch.no_grad():
        outputs2 = eab.model(
            token_tensor,
            past_key_values=cache_for_model,
            use_cache=True
        )

    logits2 = outputs2.logits[:, -1, :]
    new_cache = outputs2.past_key_values

    print(f"\nSecond token logits shape: {logits2.shape}")
    print(f"Second token logits top-5 values: {torch.topk(logits2[0], 5).values}")

    # Check if logits are reasonable
    entropy = -torch.sum(torch.nn.functional.softmax(logits2[0], dim=-1) * torch.nn.functional.log_softmax(logits2[0], dim=-1))
    normalized_entropy = entropy / torch.log(torch.tensor(eab.vocab_size, dtype=torch.float32))

    print(f"Second token entropy: {entropy.item():.4f}")
    print(f"Normalized entropy: {normalized_entropy.item():.4f}")

    if normalized_entropy.item() > 0.5:
        print(f"\n⚠️  WARNING: Very high entropy! Model is very uncertain.")
        print(f"   This suggests the cache might be corrupted or incorrectly passed.")
    else:
        print(f"\n✓ Entropy looks reasonable")

    # Check the returned cache
    if isinstance(new_cache, DynamicCache):
        new_cache_legacy = new_cache.to_legacy_cache()
        print(f"\nReturned cache: {len(new_cache_legacy)} layers")
        print(f"First layer key shape: {new_cache_legacy[0][0].shape}")

        # Extract last token
        last_key = new_cache_legacy[0][0][:, :, -1:, :]
        print(f"Last token key shape: {last_key.shape}")

        # Update COW cache
        for layer_idx, (key_full, value_full) in enumerate(new_cache_legacy):
            key_new = key_full[:, :, -1:, :]
            value_new = value_full[:, :, -1:, :]
            cow_cache.update(key_new, value_new, layer_idx)

        print(f"\nAfter update:")
        print(f"  COW cache sequence length: {cow_cache.get_seq_length()}")
        print(f"  Own cache layers: {len(cow_cache.own_cache.key_cache)}")

        # Try generating third token
        print(f"\n{'='*80}")
        print("Generating third token")
        print(f"{'='*80}")

        probs2 = torch.nn.functional.softmax(logits2[0], dim=-1)
        token_id2 = torch.multinomial(probs2, 1).item()
        token_text2 = eab.tokenizer.decode([token_id2])

        print(f"\nGenerated token: {token_id2} ('{token_text2}')")

        # Convert COW cache again
        legacy_cache3 = cow_cache.to_legacy_cache()
        print(f"Cache for third token: {len(legacy_cache3)} layers")
        print(f"First layer key shape: {legacy_cache3[0][0].shape}")

        cache_for_model3 = DynamicCache.from_legacy_cache(legacy_cache3)
        token_tensor2 = torch.tensor([[token_id2]]).to(eab.device)

        with torch.no_grad():
            outputs3 = eab.model(
                token_tensor2,
                past_key_values=cache_for_model3,
                use_cache=True
            )

        logits3 = outputs3.logits[:, -1, :]
        entropy3 = -torch.sum(torch.nn.functional.softmax(logits3[0], dim=-1) * torch.nn.functional.log_softmax(logits3[0], dim=-1))
        normalized_entropy3 = entropy3 / torch.log(torch.tensor(eab.vocab_size, dtype=torch.float32))

        print(f"\nThird token logits top-5: {torch.topk(logits3[0], 5).values}")
        print(f"Normalized entropy: {normalized_entropy3.item():.4f}")


if __name__ == "__main__":
    debug_cow_generation()
