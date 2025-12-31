"""
Generate test prompts of varying lengths for Experiment 1.A.

We create factual QA prompts by combining context from Wikipedia-style text
with questions. This ensures prompts are coherent and realistic.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import yaml


# Sample factual QA templates
CONTEXTS = [
    # Science topics
    "The theory of relativity, developed by Albert Einstein in the early 20th century, fundamentally changed our understanding of space, time, and gravity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is constant. General relativity, published in 1915, extended these concepts to include gravity as a geometric property of spacetime.",

    "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. This process occurs in chloroplasts and involves two main stages: the light-dependent reactions and the Calvin cycle. During light-dependent reactions, chlorophyll absorbs light energy to produce ATP and NADPH. The Calvin cycle then uses these molecules to fix carbon dioxide into organic compounds.",

    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens. It consists of two main components: the innate immune system, which provides immediate but non-specific defense, and the adaptive immune system, which develops targeted responses to specific pathogens. White blood cells, antibodies, and the lymphatic system play crucial roles in immune function.",

    # History topics
    "The Industrial Revolution, which began in Britain in the late 18th century, marked a major turning point in human history. It was characterized by the transition from hand production methods to machines, new chemical manufacturing processes, and the development of machine tools. The revolution led to unprecedented economic growth, urbanization, and changes in social structure. Key innovations included the steam engine, spinning jenny, and power loom.",

    "Ancient Egypt was one of the world's earliest and longest-lasting civilizations, spanning over 3000 years from around 3100 BCE to 30 BCE. The civilization was centered around the Nile River and is famous for its pyramids, hieroglyphic writing system, and pharaohs. Ancient Egyptians made significant advances in mathematics, medicine, and engineering. The society was highly organized with a complex bureaucracy and religious system.",

    # Geography topics
    "The Amazon Rainforest, located in South America, is the world's largest tropical rainforest, covering approximately 5.5 million square kilometers. It spans nine countries, with the majority in Brazil. The rainforest is home to an estimated 10% of all species on Earth and plays a crucial role in regulating global climate patterns. The Amazon River, which flows through the rainforest, is the world's largest river by volume.",

    # Technology topics
    "Artificial intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn like humans. Modern AI systems use machine learning algorithms to process large amounts of data and identify patterns. Deep learning, a subset of machine learning, uses neural networks with multiple layers to analyze data. AI applications range from natural language processing and computer vision to autonomous vehicles and medical diagnosis.",
]

QUESTIONS = [
    "What is the main topic being discussed?",
    "When did this event or discovery take place?",
    "Who was the key person or people involved?",
    "Where did this occur or what region is involved?",
    "What are the main components or features described?",
    "Why is this topic significant or important?",
    "How does this process or system work?",
    "What were the major outcomes or impacts?",
]


def get_tokenizer():
    """Get GPT-2 tokenizer for accurate token counting."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return tokenizer
    except ImportError:
        print("Warning: transformers not installed. Using approximate token counting.")
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in text."""
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    else:
        # Approximate: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)


def create_prompt(context: str, question: str, target_length: int, tokenizer=None) -> str:
    """
    Create a prompt with approximately target_length tokens.

    Args:
        context: Background context text
        question: Question to ask
        target_length: Desired prompt length in tokens
        tokenizer: Tokenizer for accurate counting

    Returns:
        Formatted prompt
    """
    # Start with basic prompt format
    base_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    current_length = count_tokens(base_prompt, tokenizer)

    # If too short, repeat or extend context
    if current_length < target_length:
        # Add more context by repeating with variations
        padding = " Furthermore, it's important to note that " + context
        while count_tokens(base_prompt + padding, tokenizer) < target_length:
            base_prompt = f"Context: {context}{padding}\n\nQuestion: {question}\n\nAnswer:"
            current_length = count_tokens(base_prompt, tokenizer)
            if current_length >= target_length:
                break
            padding += " Additionally, research has shown that " + random.choice(CONTEXTS[:3])

    # If too long, truncate context
    elif current_length > target_length * 1.1:  # Allow 10% tolerance
        words = context.split()
        while count_tokens(base_prompt, tokenizer) > target_length:
            words = words[:-10]  # Remove 10 words at a time
            context_truncated = " ".join(words)
            base_prompt = f"Context: {context_truncated}\n\nQuestion: {question}\n\nAnswer:"

    return base_prompt


def generate_prompts_for_length(
    target_length: int,
    num_prompts: int,
    tokenizer=None
) -> List[Dict[str, any]]:
    """
    Generate multiple prompts with target length.

    Args:
        target_length: Desired prompt length in tokens
        num_prompts: Number of prompts to generate
        tokenizer: Tokenizer for accurate counting

    Returns:
        List of prompt dictionaries
    """
    prompts = []
    random.seed(42 + target_length)  # Reproducible but different for each length

    for i in range(num_prompts):
        # Sample random context and question
        context = random.choice(CONTEXTS)
        question = random.choice(QUESTIONS)

        # Create prompt
        prompt_text = create_prompt(context, question, target_length, tokenizer)
        actual_length = count_tokens(prompt_text, tokenizer)

        prompts.append({
            'id': f"len{target_length}_prompt{i+1}",
            'target_length': target_length,
            'actual_length': actual_length,
            'text': prompt_text,
            'context': context[:100] + "...",  # Store truncated context for reference
            'question': question
        })

    return prompts


def main():
    """Generate all prompts for the experiment."""
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check if debug mode
    if config['debug']['enabled']:
        print("🔧 Running in DEBUG mode")
        prompt_lengths = config['debug']['prompt_lengths']
        prompts_per_length = config['debug']['prompts_per_length']
    else:
        print("🚀 Running in FULL mode")
        prompt_lengths = config['prompt_lengths']
        prompts_per_length = config['prompts_per_length']

    # Get tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    # Generate prompts for each length
    print(f"\nGenerating prompts for lengths: {prompt_lengths}")
    print(f"Prompts per length: {prompts_per_length}")
    print("-" * 60)

    for target_length in prompt_lengths:
        print(f"\nGenerating {prompts_per_length} prompts with target length {target_length} tokens...")

        prompts = generate_prompts_for_length(target_length, prompts_per_length, tokenizer)

        # Save to appropriate directory
        length_dir = Path(__file__).parent / f"length_{target_length:03d}"
        length_dir.mkdir(exist_ok=True)

        output_file = length_dir / "prompts.json"
        with open(output_file, 'w') as f:
            json.dump({
                'target_length': target_length,
                'num_prompts': len(prompts),
                'prompts': prompts
            }, f, indent=2)

        # Print summary
        actual_lengths = [p['actual_length'] for p in prompts]
        avg_length = sum(actual_lengths) / len(actual_lengths)
        print(f"  ✓ Saved {len(prompts)} prompts to {output_file}")
        print(f"  ✓ Average actual length: {avg_length:.1f} tokens (target: {target_length})")
        print(f"  ✓ Length range: {min(actual_lengths)} - {max(actual_lengths)} tokens")

    print("\n" + "=" * 60)
    print("✓ All prompts generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
