# EAB Interactive Demos

Interactive visualization and debugging tools for Entropy-Adaptive Branching.

## Recent Updates (2026-01-01)

✅ **Interactive Model Selection**: Choose from 5 models or enter custom model name:
- Llama-3.2-1B-Instruct (1B, fastest)
- Llama-3.2-3B-Instruct (3B)
- Qwen2.5-1.5B-Instruct (1.5B, fast)
- Qwen2.5-3B-Instruct (3B, default)
- Qwen2.5-7B-Instruct (7B, high quality)

✅ **Model Updated**: All demos now use **Qwen/Qwen2.5-3B-Instruct** by default (was GPT-2)
✅ **Entropy Logging Added**: Demos now extract and save entropy values for later analysis
✅ **New Outputs**:
- `demo_results/entropy_data.json` - Full entropy history with positions, values, and branch decisions
- Entropy statistics displayed in console output (mean, max, min, branch rate)
- Improved entropy plots with actual tracked values (not inferred)

## Purpose

Before running large-scale experiments, use these demos to:
1. **Verify EAB works correctly** - Check branching behavior
2. **Understand parameters** - See how threshold, branch_factor affect output
3. **Debug issues** - Visualize entropy, branching decisions, sample trees
4. **Compare with naive** - Side-by-side resource comparison

## Demos Available

### 1. `interactive_demo.py` - Main Interactive Demo ⭐

**Full-featured interactive tool** with all visualizations and comparisons.

**Features**:
- 🌳 **Sample Tree Visualization** - See branching structure
- 📊 **Entropy vs Tokens Plot** - Entropy spikes and threshold line
- 📝 **All Generated Samples** - Complete text outputs
- 🔀 **Branching Points** - Where and why branching occurred
- ⚡ **Resource Comparison** - EAB vs Naive (time, tokens, memory)
- 🎛️ **Interactive Controls** - Adjust all parameters on-the-fly

**Usage**:
```bash
# Interactive mode (prompts for all inputs)
python interactive_demo.py

# Command-line mode (specify all parameters)
python interactive_demo.py \
    --prompt "The capital of France is" \
    --threshold 0.3 \
    --branch-factor 3 \
    --max-tokens 20 \
    --max-paths 20 \
    --temperature 0.8 \
    --save-plots
```

**Output**:
- Console: Summary statistics, samples, branching info, **entropy statistics**
- Plots: `demo_results/` folder (if --save-plots)
  - `entropy_vs_tokens.png` - Entropy over generation (**with real values!**)
  - `sample_tree.png` - Branching structure
  - `resource_comparison.png` - Cost metrics
  - `all_samples.txt` - **EAB + Naive samples for comparison**
  - **`entropy_data.json`** - Full entropy history for analysis

---

### 2. `quick_test.py` - Quick Validation

**Fast sanity check** - tests EAB on predefined prompts.

**Usage**:
```bash
python quick_test.py
```

Tests:
- ✅ High confidence prompts (expect 1 sample)
- ✅ Low confidence prompts (expect many samples)
- ✅ Medium prompts (expect moderate branching)

---

### 3. `compare_thresholds.py` - Parameter Sweep

**Compare different threshold values** side-by-side.

**Usage**:
```bash
python compare_thresholds.py --prompt "Your prompt here"
```

Tests thresholds: [0.2, 0.3, 0.4, 0.5, 0.6]
Shows: Sample count, branches, cost for each

---

## Quick Start

### Option 1: Interactive Mode (Recommended)

```bash
cd /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/demos

python interactive_demo.py
```

Follow the prompts:
1. **Select a model** (1-5 for presets, 6 for custom)
   - Option 4: Qwen2.5-3B-Instruct (default, recommended)
   - Option 1: Llama-3.2-1B-Instruct (fastest for testing)
   - Option 5: Qwen2.5-7B-Instruct (best quality)
2. Enter your prompt
3. Choose threshold (or use default 0.4)
4. Choose other parameters (branch factor, max tokens, temperature)
5. Select device (CPU/CUDA)
6. View visualizations and compare with naive sampling

### Option 2: Command Line

```bash
# Default model (Qwen2.5-3B-Instruct)
python interactive_demo.py \
    --prompt "The capital of France is" \
    --threshold 0.4 \
    --max-tokens 20 \
    --save-plots

# Use Llama-3.2-1B for faster testing
python interactive_demo.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --prompt "The best programming language is" \
    --threshold 0.3 \
    --max-tokens 20 \
    --save-plots

# Use Qwen2.5-7B for best quality (slower)
python interactive_demo.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --prompt "Once upon a time" \
    --threshold 0.4 \
    --max-tokens 30 \
    --save-plots
```

---

## Understanding the Output

### 1. Sample Tree

Shows parent-child relationships between samples:
```
Root (prompt)
├─ Sample 1 (path 0)
├─ Sample 2 (path 1) ← branched at token 5
│  ├─ Sample 3 (path 2) ← branched at token 12
│  └─ Sample 4 (path 3)
└─ Sample 5 (path 4)
```

### 2. Entropy Plot

```
Entropy
  │
1.0│     *                    * = High entropy (branched)
  │    * *  *                 • = Low entropy (continued)
0.6│   *   * *    *
  │  •     • •*  •*
0.4│••           ••  •• ← Threshold line
  │                  •••
0.0└──────────────────────→ Tokens
    0  5  10 15 20 25 30
```

### 3. Resource Comparison

```
Metric          | Naive    | EAB      | Speedup
----------------|----------|----------|--------
Token-steps     | 600      | 250      | 2.4×
Wall-clock (s)  | 12.3     | 5.1      | 2.4×
Memory (MB)     | 45.2     | 18.7     | 2.4×
Samples         | 20       | 20       | Same
```

---

## Common Use Cases

### Debug: "Why isn't EAB branching?"

```bash
python interactive_demo.py \
    --prompt "Your prompt" \
    --threshold 0.3 \
    --save-plots
```

Check entropy plot:
- If entropy never exceeds threshold → model is confident
- Try lower threshold (0.2) or more uncertain prompt

### Debug: "Too many branches!"

```bash
python interactive_demo.py \
    --prompt "Your prompt" \
    --threshold 0.5 \
    --max-paths 10 \
    --save-plots
```

Increase threshold or reduce max_paths.

### Validate: "Does sample count match confidence?"

```bash
# Test high confidence
python quick_test.py

# Check: Factual prompts → 1-3 samples
# Check: Opinion prompts → 10-20 samples
```

---

## Tips

1. **Start simple**: Use interactive mode first
2. **Save plots**: Always use `--save-plots` for documentation
3. **Try extremes**: Test very low (0.2) and high (0.6) thresholds
4. **Compare prompts**: Factual vs opinion vs creative
5. **Check consistency**: Run same prompt multiple times

---

## Troubleshooting

**Problem**: No plots displayed
- **Solution**: Use `--save-plots` flag, check `demo_results/` folder

**Problem**: All samples identical
- **Solution**: Model is confident, try more uncertain prompt

**Problem**: Too slow
- **Solution**: Reduce `--max-tokens` or use CPU

**Problem**: Entropy always low
- **Solution**: Check temperature (should be > 0.5), try creative prompts

---

## Files

- `interactive_demo.py` - Main demo (full features)
- `quick_test.py` - Fast validation
- `compare_thresholds.py` - Parameter sweep
- `utils.py` - Shared visualization functions
- `demo_results/` - Output folder (created automatically)

---

**Last Updated**: 2026-01-01
**Status**: Ready to use
