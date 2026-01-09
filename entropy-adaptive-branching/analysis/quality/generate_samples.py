"""
Generate real EAB and Naive samples for quality assessment.

This script:
1. Creates 100 diverse prompts (open-ended for creativity)
2. Generates samples using EAB (entropy-adaptive branching)
3. Generates samples using Naive (varied temperature)
4. Saves to JSON for quality_assessment.py
"""

import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add EAB to path
eab_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(eab_root))

from eab import EntropyAdaptiveBranching
from transformers import AutoModelForCausalLM, AutoTokenizer

quality_dir = Path(__file__).parent


# =============================================================================
# PROMPT GENERATION
# =============================================================================

OPEN_ENDED_PROMPTS = [
    # Opinion/Perspective
    "The best way to learn a new skill is",
    "The most important quality in a leader is",
    "The biggest challenge facing society today is",
    "The key to a successful relationship is",
    "The most valuable lesson I've learned is",

    # How-to/Advice
    "How can we reduce climate change",
    "How can we improve education systems",
    "How can we make cities more sustainable",
    "How can we promote mental health",
    "How can we encourage innovation",

    # What are
    "What are the benefits of exercise",
    "What are the advantages of renewable energy",
    "What are the keys to effective communication",
    "What are the characteristics of a good friend",
    "What are the elements of a great story",

    # Descriptive
    "The future of technology will be",
    "The ideal workplace should",
    "A perfect day would include",
    "The role of art in society is",
    "The meaning of success is",

    # Comparative
    "The difference between knowledge and wisdom is",
    "The balance between work and life requires",
    "The relationship between nature and humanity is",
    "The connection between creativity and science is",
    "The distinction between happiness and contentment is",

    # Conditional/Hypothetical
    "If I could change one thing about the world, it would be",
    "If everyone learned one skill, it should be",
    "If we could solve one problem, it should be",
    "If technology continues advancing, we will",
    "If education focused on one thing, it should be",

    # Exploratory
    "The reasons people travel are",
    "The impact of social media has been",
    "The evolution of communication shows that",
    "The importance of diversity stems from",
    "The power of storytelling lies in",

    # Philosophical
    "The purpose of education is",
    "The foundation of trust is",
    "The essence of creativity is",
    "The nature of intelligence is",
    "The value of failure is",

    # Problem-solving
    "To build a better community, we should",
    "To address inequality, we need to",
    "To preserve the environment, we must",
    "To foster innovation, organizations should",
    "To improve healthcare, we could",

    # Narrative starters
    "The most interesting person I know is someone who",
    "A life-changing experience happened when",
    "The greatest discovery in history was",
    "An unexpected lesson came from",
    "The turning point in my thinking occurred when",
]

# Generate more variations
def generate_prompt_variations() -> List[str]:
    """Create 100+ diverse prompts."""
    prompts = OPEN_ENDED_PROMPTS.copy()

    # Add more variations
    topics = [
        "reading", "music", "cooking", "gardening", "coding",
        "painting", "writing", "meditation", "volunteering", "traveling",
        "friendship", "family", "community", "culture", "tradition",
        "science", "history", "philosophy", "psychology", "economics"
    ]

    templates = [
        "The beauty of {} lies in",
        "The challenge of {} is that",
        "The appeal of {} comes from",
        "The art of {} requires",
        "The practice of {} teaches us",
        "Why {} matters for",
        "When {} becomes important",
        "Where {} can lead us",
    ]

    for topic in topics[:15]:
        template = random.choice(templates)
        prompts.append(template.format(topic))

    # Ensure we have at least 100 unique prompts
    prompts = list(set(prompts))[:100]

    return prompts


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_eab_samples(
    prompts: List[str],
    model_name: str = "gpt2",
    entropy_threshold: float = 0.4,
    branch_factor: int = 3,
    max_paths: int = 20,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    device: str = "cuda"
) -> Dict[str, List[str]]:
    """
    Generate samples using EAB method.

    Args:
        prompts: List of prompts
        model_name: Model to use
        entropy_threshold: When to branch
        branch_factor: How many branches
        max_paths: Max concurrent paths
        max_new_tokens: Generation length
        temperature: Fixed temperature
        device: Device to use

    Returns:
        Dictionary {prompt: [generations]}
    """
    print("\n" + "=" * 70)
    print("GENERATING EAB SAMPLES")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Entropy threshold: {entropy_threshold}")
    print(f"Branch factor: {branch_factor}")
    print(f"Max paths: {max_paths}")
    print(f"Temperature: {temperature} (fixed)")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Initialize EAB
    eab = EntropyAdaptiveBranching(
        model_name=model_name,
        entropy_threshold=entropy_threshold,
        branch_factor=branch_factor,
        max_paths=max_paths,
        device=device
    )

    results = {}

    for prompt in tqdm(prompts, desc="EAB Generation"):
        try:
            # Generate with EAB
            outputs = eab.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_metadata=True,
                show_progress=False,
                use_chat_template=False  # For open-ended generation
            )

            # Extract text continuations (remove original prompt)
            generations = []
            for output in outputs:
                full_text = output['text']
                # Remove prompt to get only the generation
                if full_text.startswith(prompt):
                    continuation = full_text[len(prompt):].strip()
                else:
                    continuation = full_text.strip()

                if continuation:  # Only add non-empty
                    generations.append(continuation)

            results[prompt] = generations

        except Exception as e:
            print(f"\n⚠ Error with prompt '{prompt[:50]}...': {e}")
            results[prompt] = []

    # Print statistics
    print("\n" + "-" * 70)
    print("EAB GENERATION STATISTICS")
    print("-" * 70)
    counts = [len(gens) for gens in results.values()]
    print(f"Total prompts: {len(results)}")
    print(f"Avg samples per prompt: {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    print(f"Min/Max samples: {min(counts)} / {max(counts)}")
    print("-" * 70 + "\n")

    return results


def generate_naive_samples(
    prompts: List[str],
    target_samples_dict: Dict[str, int],
    model_name: str = "gpt2",
    temp_range: tuple = (0.7, 1.3),
    max_new_tokens: int = 30,
    device: str = "cuda"
) -> Dict[str, List[str]]:
    """
    Generate samples using naive varied-temperature baseline.

    Args:
        prompts: List of prompts
        target_samples_dict: How many samples per prompt (match EAB)
        model_name: Model to use
        temp_range: Temperature range for diversity
        max_new_tokens: Generation length
        device: Device to use

    Returns:
        Dictionary {prompt: [generations]}
    """
    print("\n" + "=" * 70)
    print("GENERATING NAIVE SAMPLES (Varied Temperature)")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Temperature range: {temp_range}")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for prompt in tqdm(prompts, desc="Naive Generation"):
        num_samples = target_samples_dict.get(prompt, 20)
        generations = []

        try:
            # Generate with varied temperature
            for _ in range(num_samples):
                # Sample random temperature for diversity
                temp = random.uniform(*temp_range)

                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decode
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove prompt
                if full_text.startswith(prompt):
                    continuation = full_text[len(prompt):].strip()
                else:
                    continuation = full_text.strip()

                if continuation:
                    generations.append(continuation)

            results[prompt] = generations

        except Exception as e:
            print(f"\n⚠ Error with prompt '{prompt[:50]}...': {e}")
            results[prompt] = []

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Print statistics
    print("\n" + "-" * 70)
    print("NAIVE GENERATION STATISTICS")
    print("-" * 70)
    counts = [len(gens) for gens in results.values()]
    print(f"Total prompts: {len(results)}")
    print(f"Avg samples per prompt: {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    print(f"Min/Max samples: {min(counts)} / {max(counts)}")
    print("-" * 70 + "\n")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main generation workflow."""
    print("=" * 70)
    print("QUALITY SAMPLE GENERATION")
    print("=" * 70)

    # Configuration
    MODEL_NAME = "gpt2"  # Start with smaller model, change to gpt2-medium/large if needed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_PROMPTS = 100
    MAX_NEW_TOKENS = 30

    # EAB settings
    EAB_TEMP = 1.0
    EAB_ENTROPY_THRESHOLD = 0.4
    EAB_BRANCH_FACTOR = 3
    EAB_MAX_PATHS = 20

    # Naive settings
    NAIVE_TEMP_RANGE = (0.7, 1.3)

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Num prompts: {NUM_PROMPTS}")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  EAB temperature: {EAB_TEMP} (fixed)")
    print(f"  Naive temperature: {NAIVE_TEMP_RANGE} (varied)")

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # -------------------------------------------------------------------------
    # 1. Generate prompts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Generating Prompts")
    print("=" * 70)

    prompts = generate_prompt_variations()
    print(f"✓ Generated {len(prompts)} unique prompts")

    # Save prompts for reference
    prompts_file = quality_dir / "prompts_used.json"
    with open(prompts_file, 'w') as f:
        json.dump({"prompts": prompts}, f, indent=2)
    print(f"✓ Saved to {prompts_file}")

    # -------------------------------------------------------------------------
    # 2. Generate EAB samples
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Generating EAB Samples")
    print("=" * 70)

    eab_samples = generate_eab_samples(
        prompts=prompts,
        model_name=MODEL_NAME,
        entropy_threshold=EAB_ENTROPY_THRESHOLD,
        branch_factor=EAB_BRANCH_FACTOR,
        max_paths=EAB_MAX_PATHS,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=EAB_TEMP,
        device=DEVICE
    )

    # Save EAB samples
    eab_file = quality_dir / "eab_samples.json"
    with open(eab_file, 'w') as f:
        json.dump(eab_samples, f, indent=2)
    print(f"✓ Saved EAB samples to {eab_file}")

    # -------------------------------------------------------------------------
    # 3. Generate Naive samples (match EAB sample counts)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Generating Naive Samples")
    print("=" * 70)

    # Create target counts dict (match EAB)
    target_counts = {prompt: len(samples) for prompt, samples in eab_samples.items()}

    naive_samples = generate_naive_samples(
        prompts=prompts,
        target_samples_dict=target_counts,
        model_name=MODEL_NAME,
        temp_range=NAIVE_TEMP_RANGE,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE
    )

    # Save Naive samples
    naive_file = quality_dir / "naive_samples.json"
    with open(naive_file, 'w') as f:
        json.dump(naive_samples, f, indent=2)
    print(f"✓ Saved Naive samples to {naive_file}")

    # -------------------------------------------------------------------------
    # 4. Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print(f"  • {prompts_file.name} - All prompts used")
    print(f"  • {eab_file.name} - EAB generations")
    print(f"  • {naive_file.name} - Naive generations")

    print("\nSample counts comparison:")
    for prompt in prompts[:5]:
        eab_count = len(eab_samples.get(prompt, []))
        naive_count = len(naive_samples.get(prompt, []))
        print(f"  '{prompt[:40]}...'")
        print(f"    EAB: {eab_count}, Naive: {naive_count}")

    print("\n" + "=" * 70)
    print("Next step:")
    print("  Run: python quality_assessment.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
