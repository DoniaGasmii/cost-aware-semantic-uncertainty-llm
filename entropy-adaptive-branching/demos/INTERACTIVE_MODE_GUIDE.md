# Interactive Mode Guide

Complete walkthrough of the interactive demo with model selection.

## Starting the Demo

```bash
cd /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/demos
python3 interactive_demo.py
```

## Interactive Flow

### Step 1: Model Selection

```
============================================================
  EAB Interactive Demo - Verify Branching Behavior
============================================================

Select a model:
  1. Llama-3.2-1B-Instruct (1B params, fastest)
  2. Llama-3.2-3B-Instruct (3B params)
  3. Qwen2.5-1.5B-Instruct (1.5B params, fast)
  4. Qwen2.5-3B-Instruct (3B params, default)
  5. Qwen2.5-7B-Instruct (7B params, high quality)
  6. Custom (enter model name manually)

Choice (1-6, default: 4):
>
```

**What to choose**:
- **Option 1 (Llama-3.2-1B)**: Fastest for quick testing and iteration
- **Option 4 (Qwen2.5-3B)**: Recommended - good balance of speed and quality
- **Option 5 (Qwen2.5-7B)**: Best quality but slower (use for final validation)
- **Option 6 (Custom)**: Advanced - enter any HuggingFace model name

**Example**: Press Enter for default (option 4)

---

### Step 2: Prompt Input

```
Enter your prompt (or press Enter for default):
>
```

**Examples**:
- Factual: `What is the capital of Tunisia?`
- Creative: `Once upon a time in a magical forest`
- Opinion: `The best programming language is`
- Math: `2 + 2 =`

**Example**: Type `What is the capital of Tunisia?` and press Enter

---

### Step 3: Entropy Threshold

```
Entropy threshold (0.0-1.0, default: 0.4):
>
```

**Guide**:
- **0.2-0.3**: More branching, more samples (use for uncertain prompts)
- **0.4**: Balanced (recommended default)
- **0.5-0.6**: Less branching, fewer samples (use for confident prompts)

**Example**: Press Enter for default (0.4)

---

### Step 4: Branch Factor

```
Branch factor (how many paths to create, default: 3):
>
```

**Guide**:
- **2**: Minimal branching (faster)
- **3**: Balanced (recommended)
- **5**: More exploration (slower)

**Example**: Press Enter for default (3)

---

### Step 5: Max Tokens

```
Max new tokens to generate (default: 20):
>
```

**Guide**:
- **10-20**: Quick testing
- **30-50**: Standard generation
- **100+**: Long-form text (slow)

**Example**: Type `30` and press Enter

---

### Step 6: Max Paths

```
Max total paths (default: 20):
>
```

**Guide**:
- **10**: Faster, less diversity
- **20**: Balanced (recommended)
- **50**: Maximum diversity (slower)

**Example**: Press Enter for default (20)

---

### Step 7: Temperature

```
Temperature (0.0-2.0, default: 0.8):
>
```

**Guide**:
- **0.7-0.8**: Balanced randomness (recommended)
- **0.5-0.6**: More focused
- **0.9-1.0**: More random

**Example**: Press Enter for default (0.8)

---

### Step 8: Device Selection

```
Device (cpu/cuda, default: cpu):
>
```

**Guide**:
- **cpu**: Works everywhere (slower)
- **cuda**: GPU acceleration (faster, requires NVIDIA GPU)

**Example**: Type `cpu` and press Enter

---

### Step 9: Save Plots

```
Save plots to demo_results/? (y/n, default: y):
>
```

**Guide**:
- **y**: Save all plots and data to disk (recommended)
- **n**: Only display plots (not saved)

**Example**: Press Enter for default (y)

---

## Output

After all prompts, the demo will:

1. **Load the model** (may take 10-30 seconds)
2. **Generate with EAB** and show progress
3. **Generate with Naive** for comparison
4. **Display summary** with statistics including:
   ```
   --- EAB Results ---
   Samples generated: 13
   Total branches: 4
   Branch positions: [9, 16, 22, 23]
   Total tokens: 390
   Wall time: 15.32s
   Peak memory: 1234.5 MB

   Entropy Statistics:
     Mean entropy: 0.342
     Max entropy: 0.587
     Min entropy: 0.123
     Branch rate: 11.3%

   --- Naive Results ---
   Samples generated: 13
   Total tokens: 390
   Wall time: 18.45s
   Peak memory: 1450.2 MB

   --- Comparison ---
   Speedup: 1.20×
   Token reduction: 1.00×
   Memory reduction: 1.17×
   ```

5. **Create visualizations**:
   - Entropy vs tokens plot (with real entropy values!)
   - Sample tree structure
   - Resource comparison bar chart

6. **Save to disk** (if you chose yes):
   ```
   demo_results/
   ├── entropy_vs_tokens.png
   ├── sample_tree.png
   ├── resource_comparison.png
   ├── all_samples.txt
   └── entropy_data.json
   ```

---

## Quick Presets

For common use cases, you can use these settings:

### Preset 1: Quick Test (Fastest)
- Model: **1** (Llama-3.2-1B)
- Prompt: `What is 2 + 2?`
- Threshold: **0.4**
- Tokens: **10**
- Everything else: defaults

**Use when**: Testing if EAB works, debugging code

---

### Preset 2: Standard Demo (Recommended)
- Model: **4** (Qwen2.5-3B) - default
- Prompt: `What is the capital of Tunisia?`
- Threshold: **0.4**
- Tokens: **30**
- Everything else: defaults

**Use when**: Normal demonstration, showing to others

---

### Preset 3: High Quality (Slower)
- Model: **5** (Qwen2.5-7B)
- Prompt: `Once upon a time in a magical forest`
- Threshold: **0.3** (more branching for creative text)
- Tokens: **50**
- Temperature: **0.9** (more creative)
- Everything else: defaults

**Use when**: Final validation, publication-quality results

---

## Tips

1. **First run is slower**: Model loading takes time, subsequent runs are faster
2. **Save plots**: Always use yes to save - helps with debugging later
3. **Start small**: Use 1B model and 10 tokens for initial testing
4. **Check entropy_data.json**: Contains all entropy values for analysis
5. **Compare models**: Run same prompt with different models (1, 4, 5) to compare

---

## Command-Line Shortcuts

Skip interactive mode by providing all parameters:

```bash
# Quick test with Llama-1B
python3 interactive_demo.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --prompt "What is 2+2?" \
    --threshold 0.4 \
    --max-tokens 10 \
    --save-plots

# Standard demo with Qwen-3B (default)
python3 interactive_demo.py \
    --prompt "What is the capital of Tunisia?" \
    --threshold 0.4 \
    --max-tokens 30 \
    --save-plots

# High quality with Qwen-7B
python3 interactive_demo.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --prompt "Once upon a time" \
    --threshold 0.3 \
    --max-tokens 50 \
    --temperature 0.9 \
    --save-plots
```

---

**Happy exploring!** 🎉
