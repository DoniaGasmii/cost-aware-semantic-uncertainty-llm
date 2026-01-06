# EAB Quality Guide: Getting Coherent, Branching Demos

This guide helps you achieve high-quality, coherent generation with observable branching behavior.

## Quick Fixes

### 1. **CRITICAL: Use Chat Template** ✅
Always use `use_chat_template=True` (now the default) for instruct models:

```python
eab.generate(
    prompt="What is the best approach to learn Python?",
    use_chat_template=True  # Default, but be explicit
)
```

**Why**: Instruct models need special formatting to understand they're answering questions, not just continuing text.

### 2. **Choose Uncertainty-Inducing Prompts**

**❌ Bad prompts (too deterministic, no branching):**
- "2 + 2 ="
- "The capital of France is"
- "Water boils at"

**✅ Good prompts (high uncertainty, will branch):**
- "What is the best way to learn programming?"
- "Write a creative opening for a sci-fi story"
- "What will technology look like in 2050?"
- "Explain the pros and cons of remote work"
- "Complete this story: Once upon a time in a distant galaxy..."

**Why**: EAB only branches when entropy is high. Factual questions have low entropy (model is confident), so you won't see branching behavior.

### 3. **Adjust Generation Parameters**

```python
eab.generate(
    prompt="Your creative prompt here",
    max_new_tokens=50,           # Longer = more branching opportunities
    temperature=0.8,             # 0.7-0.9 for good diversity
    entropy_threshold=0.15,      # Lower = more branching (try 0.1-0.2)
    branch_factor=2,             # Start with 2, increase to 3-4 for more diversity
    max_paths=15,                # More paths = more samples
)
```

**Parameter Guide:**
- **temperature**:
  - Too low (< 0.5): Very confident, little diversity
  - Good (0.7-0.9): Balanced confidence and diversity
  - Too high (> 1.2): Random, incoherent

- **entropy_threshold**:
  - Low (0.1-0.15): Frequent branching, high exploration
  - Medium (0.2-0.25): Balanced
  - High (0.3+): Rare branching, mostly single path

- **max_new_tokens**:
  - Short (10-20): Good for testing, fewer branch opportunities
  - Medium (30-50): Recommended for demos
  - Long (100+): More branching opportunities but slower

## Model Selection Strategy

### Option 1: Current 3B Model (Recommended for 24GB GPU)
```python
eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-3B-Instruct',
    device='cuda',
    torch_dtype=torch.float16  # ~6GB VRAM
)
```
**Pros**: Fast, fits easily, good quality
**Cons**: Less capable than larger models

### Option 2: 7B Model with FP16 (Fits on 24GB)
```python
eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-7B-Instruct',
    device='cuda',
    torch_dtype=torch.float16  # ~14GB VRAM
)
```
**Pros**: Better quality, more coherent
**Cons**: Uses more memory, slower

### Option 3: 7B Model with 8-bit Quantization (Memory-Efficient)
```python
eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-7B-Instruct',
    device='cuda',
    load_in_8bit=True  # ~7GB VRAM (requires bitsandbytes)
)
```
**Pros**: Best quality-to-memory ratio
**Cons**: Requires `bitsandbytes` library, slight quality drop

**Installation for 8-bit:**
```bash
pip install bitsandbytes accelerate
```

### Option 4: Smaller 1.5B Model (If still OOM)
```python
eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-1.5B-Instruct',
    device='cuda',
    torch_dtype=torch.float16  # ~3GB VRAM
)
```

## Expected Behavior by Prompt Type

### High Confidence (Low Entropy, Little/No Branching)
```python
# Example: Factual question
samples = eab.generate("What is the capital of France?")
# Expected: 1-3 samples, 0-1 branches, average entropy < 0.1
```

### Medium Confidence (Moderate Branching)
```python
# Example: Opinion/approach question
samples = eab.generate("What is the best way to learn programming?")
# Expected: 5-12 samples, 3-8 branches, average entropy 0.15-0.25
```

### Low Confidence (High Branching)
```python
# Example: Creative prompt
samples = eab.generate("Write the opening to a mystery novel set in space")
# Expected: 10-20 samples, 8-15 branches, average entropy 0.2-0.4
```

## Testing for Quality

### Quick Quality Test
```bash
cd demos
python3 test_chat_template.py
```

### Interactive Testing
```bash
cd demos
python3 interactive_demo.py
```

Try these test prompts:
1. **Creative**: "In the year 2100, humans will..."
2. **Opinion**: "The most important skill for future careers is..."
3. **Story**: "Once upon a time, in a world where magic and technology merged..."
4. **Advice**: "To become a better programmer, you should..."

### What Good Output Looks Like

**✅ Coherent branching:**
```
Branch at position 5 (entropy: 0.23)
  Path 1: "...you should practice daily by building projects..."
  Path 2: "...you should read other people's code to learn..."
  Path 3: "...you should focus on understanding fundamentals..."
```

**❌ Random/incoherent:**
```
Branch at position 5 (entropy: 0.21)
  Path 1: "...asdfkjh random tokens here..."
  Path 2: "...!!!! ???? ..."
```

If you see random tokens, check:
1. ✅ Using chat template?
2. ✅ Temperature not too high (< 1.0)?
3. ✅ Using an instruct model?
4. ✅ Prompt is in English and well-formed?

## Memory Optimization Tips

If you hit OOM errors:

1. **Reduce max_paths**: `max_paths=10` (from 20)
2. **Reduce branch_factor**: `branch_factor=2` (from 3)
3. **Use FP16**: `torch_dtype=torch.float16`
4. **Use 8-bit**: `load_in_8bit=True`
5. **Clear GPU before running**:
   ```bash
   pkill -f python3  # Kill other Python processes
   python3 your_script.py
   ```
6. **Use smaller model**: Qwen2.5-1.5B or Qwen2.5-3B

## Recommended Demo Configuration

For a **coherent, visually impressive demo** on 24GB GPU:

```python
import torch
from eab import EntropyAdaptiveBranching

eab = EntropyAdaptiveBranching(
    model_name='Qwen/Qwen2.5-3B-Instruct',  # or 7B if memory allows
    entropy_threshold=0.15,                  # Moderate branching
    branch_factor=3,                         # 3-way branches
    max_paths=15,                            # Good sample diversity
    device='cuda',
    torch_dtype=torch.float16
)

samples = eab.generate(
    prompt="What will technology look like in 2050?",  # Uncertain, creative
    max_new_tokens=50,
    temperature=0.8,
    use_chat_template=True  # CRITICAL
)

for i, sample in enumerate(samples, 1):
    print(f"\n{i}. {sample['generated_only']}")
    print(f"   Branches at: {sample['branch_points']}")
```

## Running Experiments

Once you have coherent demos, run your experiments with varied prompts:

```python
# Create test suite with different confidence levels
test_prompts = {
    'high_confidence': [
        "What is the capital of France?",
        "2 + 2 = ",
    ],
    'medium_confidence': [
        "What is the best way to learn programming?",
        "Explain the advantages of renewable energy",
    ],
    'low_confidence': [
        "Write a creative story about time travel",
        "What will the world look like in 2100?",
    ]
}

for category, prompts in test_prompts.items():
    for prompt in prompts:
        samples = eab.generate(prompt, max_new_tokens=50, temperature=0.8)
        # Measure: memory, time, branching, diversity...
```

## Summary Checklist

Before running full experiments, verify:

- [ ] ✅ Chat template enabled (`use_chat_template=True`)
- [ ] ✅ Using instruct model (e.g., Qwen2.5-3B-Instruct)
- [ ] ✅ Test with creative/uncertain prompts (not just factual)
- [ ] ✅ Temperature 0.7-0.9
- [ ] ✅ Entropy threshold 0.15-0.2
- [ ] ✅ FP16 enabled for memory efficiency
- [ ] ✅ Outputs are coherent and relevant to prompt
- [ ] ✅ Observable branching on uncertain prompts (check branch_points)

Once all checked, you're ready for experiments! 🚀
