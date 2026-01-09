"""
Generate test prompts of fixed length (200 tokens) for experiments.
Uses Qwen tokenizer for accurate counting.
"""

import json
import random
from pathlib import Path
import yaml


CONTEXTS = [
    "The theory of relativity, developed by Albert Einstein in the early 20th century, fundamentally changed our understanding of space, time, and gravity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is constant.",

    "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs in chloroplasts and involves two main stages: the light-dependent reactions and the Calvin cycle.",

    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens. It consists of two main components: the innate immune system and the adaptive immune system.",

    "The Industrial Revolution began in Britain in the late 18th century and marked a major turning point in human history. It was characterized by the transition from hand production methods to machines and new chemical manufacturing processes.",

    "Ancient Egypt was one of the world's earliest and longest-lasting civilizations, spanning over 3000 years. The civilization was centered around the Nile River and is famous for its pyramids and hieroglyphic writing system.",

    "The Amazon Rainforest is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers across nine countries. It is home to an estimated 10% of all species on Earth.",

    "Artificial intelligence refers to the simulation of human intelligence in machines programmed to think and learn. Modern AI systems use machine learning algorithms to process large amounts of data and identify patterns.",

    "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the Industrial Revolution, human activities have been the main driver of climate change, primarily due to burning fossil fuels.",

    "The Internet revolutionized global communication and information sharing. Developed from military research in the 1960s, it has become an essential part of modern life, connecting billions of people worldwide.",

    "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales of energy levels of atoms and subatomic particles. It challenges our classical understanding of how physical systems behave.",
]

QUESTIONS = [
    "What is the main topic being discussed?",
    "When did this event or discovery take place?",
    "What are the key features described?",
    "Why is this topic significant?",
    "How does this process work?",
    "What were the major outcomes?",
    "Who was involved in this development?",
    "Where did this occur?",
]


def get_tokenizer():
    """Get Qwen tokenizer for accurate token counting."""
    from transformers import AutoTokenizer
    # Load from parent config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    return tokenizer


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text))


def create_prompt(context: str, question: str, target_length: int, tokenizer) -> str:
    """
    Create a prompt with approximately target_length tokens.

    Format: Context: ...\n\nQuestion: ...\n\nAnswer:
    """
    base_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    current_length = count_tokens(base_prompt, tokenizer)

    # If too short, add more context
    if current_length < target_length:
        padding = " Furthermore, " + context
        while count_tokens(base_prompt + padding, tokenizer) < target_length:
            base_prompt = f"Context: {context}{padding}\n\nQuestion: {question}\n\nAnswer:"
            current_length = count_tokens(base_prompt, tokenizer)
            if current_length >= target_length:
                break
            padding += " Additionally, " + random.choice(CONTEXTS[:3])

    # If too long, truncate context
    elif current_length > target_length * 1.1:
        words = context.split()
        while count_tokens(base_prompt, tokenizer) > target_length:
            words = words[:-5]
            context = " ".join(words)
            base_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

    return base_prompt


def generate_prompts(target_length: int, num_prompts: int, tokenizer):
    """Generate prompts of target length."""
    prompts = []

    for i in range(num_prompts):
        # Randomly select context and question
        context = random.choice(CONTEXTS)
        question = random.choice(QUESTIONS)

        prompt_text = create_prompt(context, question, target_length, tokenizer)
        actual_length = count_tokens(prompt_text, tokenizer)

        prompts.append({
            'id': f"prompt_{target_length:03d}_{i+1:02d}",
            'text': prompt_text,
            'target_length': target_length,
            'actual_length': actual_length,
            'context_source': CONTEXTS.index(context),
            'question': question
        })

    return prompts


def main():
    print("=" * 70)
    print("GENERATING PROMPTS")
    print("=" * 70)

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"   ✓ Using tokenizer: {config['model']['name']}")

    # Determine mode
    if config['debug']['enabled']:
        print("\n   🔧 DEBUG MODE")
    else:
        print("\n   🚀 FULL MODE")

    # Generate prompts for fixed length
    target_length = config['prompt_length']
    num_prompts = 20  # Generate extra, use subset based on mode

    print(f"\n2. Generating prompts...")
    print(f"   Target length: {target_length} tokens")
    print(f"   Number of prompts: {num_prompts}")

    random.seed(config['seed'])
    prompts = generate_prompts(target_length, num_prompts, tokenizer)

    # Save prompts
    prompts_dir = Path(__file__).parent / f"length_{target_length:03d}"
    prompts_dir.mkdir(exist_ok=True, parents=True)

    output_file = prompts_dir / "prompts.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'target_length': target_length,
                'num_prompts': len(prompts),
                'seed': config['seed'],
                'model': config['model']['name']
            },
            'prompts': prompts
        }, f, indent=2)

    print(f"\n   ✓ Saved {len(prompts)} prompts to: {output_file}")

    # Print sample
    print(f"\n3. Sample prompt (truncated):")
    print("-" * 70)
    sample = prompts[0]
    print(f"ID: {sample['id']}")
    print(f"Length: {sample['actual_length']} tokens (target: {sample['target_length']})")
    print(f"Text: {sample['text'][:200]}...")
    print("-" * 70)

    print("\n✅ PROMPT GENERATION COMPLETE")


if __name__ == "__main__":
    main()
