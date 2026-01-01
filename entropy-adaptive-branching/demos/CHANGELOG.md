# Demo System Changelog

## 2026-01-01 (Update 4) - Sample Tree Plot Fix

### Changes Made

#### Fixed Sample Tree Visualization

**Affected Files**:
- [utils.py](utils.py:150-183) - Fixed root node detection
- [utils.py](utils.py:234-261) - Handle virtual root nodes
- [utils.py](utils.py:263-268) - Updated legend

**Problem**:
Sample tree plot was empty/not showing anything when all samples had a parent_id set (no explicit root with `parent_id=None`).

**Root Cause**:
EAB samples reference a virtual "root" node (e.g., `parent_id=0` representing the initial prompt state), but this root node doesn't exist as an actual sample. The plotting code looked for nodes with `parent_id=None`, found none, and failed to layout the tree.

**Solution**:
1. **Detect virtual roots**: Find parent_ids that don't correspond to any path_id (e.g., node 0 is a parent but not a sample)
2. **Add virtual roots to tree layout**: Use these as starting points for the tree layout
3. **Render virtual nodes**: Draw virtual root nodes as blue squares labeled "Root"
4. **Updated legend**: Added entry for root nodes

**Before**: Blank plot or "No branching occurred" message
**After**: Proper tree visualization with root node and all branches

**Example Output**:
```
Root (blue square)
  ├─ P2 (green circle, 1 branch)
  ├─ P7 (yellow circle, 3 branches)
  │   ├─ P11 (...)
  │   └─ P13 (...)
  └─ P5 (orange circle, 4+ branches)
```

---

## 2026-01-01 (Update 3) - Naive Samples in Output File

### Changes Made

#### Save Both EAB and Naive Samples for Comparison

**Affected Files**:
- [utils.py](utils.py:305-353) - Updated save_samples_to_file function
- [interactive_demo.py](interactive_demo.py:427-434) - Pass naive samples to save function
- [README.md](README.md:65) - Updated documentation

**What Changed**:

The `all_samples.txt` output file now includes **both EAB and Naive samples** for easy side-by-side comparison.

**New Output Format**:
```
======================================================================
  Generated Samples Comparison
======================================================================

Prompt: What is the capital of Tunisia?

======================================================================
  EAB SAMPLES (13 total)
======================================================================

[EAB Sample 1] Path 2 (parent: 0)
  Branches: 1 at positions [9]
  Text: ...
----------------------------------------------------------------------

... (all EAB samples)

======================================================================
  NAIVE SAMPLES (13 total)
======================================================================

[Naive Sample 1] (30 tokens)
  Text: ...
----------------------------------------------------------------------

... (all Naive samples)

======================================================================
  End of Comparison
======================================================================
```

**Benefits**:
- Easy comparison of EAB vs Naive outputs
- See if EAB explores more diverse completions
- Verify both methods generate similar quality
- All data in one file for convenience

**Backward Compatibility**:
- Function signature changed to accept optional `naive_samples` parameter
- If `naive_samples=None`, only EAB samples are saved (legacy behavior)

---

## 2026-01-01 (Update 2) - Interactive Model Selection

### Changes Made

#### Interactive Model Selection Menu

**Affected Files**:
- [interactive_demo.py](interactive_demo.py:59-90) - Added model selection menu
- [interactive_demo.py](interactive_demo.py:124-131) - Added device selection
- [README.md](README.md:7-12) - Documented new feature

**New Feature**:

When running in interactive mode, users now see a menu to select from 5 preset models or enter a custom model:

```
Select a model:
  1. Llama-3.2-1B-Instruct (1B params, fastest)
  2. Llama-3.2-3B-Instruct (3B params)
  3. Qwen2.5-1.5B-Instruct (1.5B params, fast)
  4. Qwen2.5-3B-Instruct (3B params, default)
  5. Qwen2.5-7B-Instruct (7B params, high quality)
  6. Custom (enter model name manually)

Choice (1-6, default: 4):
```

**Benefits**:
- Easy switching between models for comparison
- No need to remember exact HuggingFace model names
- Size/quality tradeoff clearly labeled
- Custom model option for flexibility

**Usage**:
```bash
# Interactive mode - select model from menu
python3 interactive_demo.py

# Command-line mode - specify model directly
python3 interactive_demo.py --model "meta-llama/Llama-3.2-1B-Instruct" --prompt "Hello"
```

---

## 2026-01-01 (Update 1) - Model Update + Entropy Logging

### Changes Made

#### 1. Model Updated to Qwen/Qwen2.5-3B-Instruct

**Affected Files**:
- [interactive_demo.py](interactive_demo.py:238)
- [quick_test.py](quick_test.py:146)
- [compare_thresholds.py](compare_thresholds.py:37)

**Before**: `model_name='gpt2'` (124M parameters, old 2019 model)
**After**: `model_name='Qwen/Qwen2.5-3B-Instruct'` (3B parameters, modern 2024 model)

**Why**:
- GPT-2 is outdated and produces low-quality outputs
- Qwen 2.5 is more recent, multilingual, and instruction-tuned
- Better quality for validating EAB behavior
- Still small enough to run on CPU

---

#### 2. Entropy Logging Added

**Affected Files**:
- [interactive_demo.py](interactive_demo.py:307) - Extract entropy history
- [interactive_demo.py](interactive_demo.py:325) - Store in metrics
- [interactive_demo.py](interactive_demo.py:182-188) - Display stats
- [interactive_demo.py](interactive_demo.py:384-396) - Save to JSON
- [utils.py](utils.py:13-58) - Updated plot function

**New Functionality**:

1. **Entropy Extraction** (line 307):
   ```python
   entropy_data = eab.get_entropy_history()
   ```
   Calls EAB's built-in method to get:
   - `positions`: Token positions [0, 1, 2, ...]
   - `entropies`: Entropy value at each position [0.25, 0.48, ...]
   - `branched`: Boolean for each position (did branching occur?)
   - `statistics`: Mean, max, min entropy, branch rate

2. **Console Display** (lines 182-188):
   ```
   Entropy Statistics:
     Mean entropy: 0.342
     Max entropy: 0.587
     Min entropy: 0.123
     Branch rate: 11.3%
   ```

3. **JSON Export** (lines 384-396):
   ```json
   {
     "positions": [0, 1, 2, 3, ...],
     "entropies": [0.25, 0.31, 0.48, 0.52, ...],
     "branched": [false, false, true, false, ...],
     "statistics": {
       "mean_entropy": 0.342,
       "max_entropy": 0.587,
       "min_entropy": 0.123,
       "branch_rate": 0.113
     },
     "threshold": 0.4,
     "prompt": "Your prompt here"
   }
   ```

   Saved to: `demo_results/entropy_data.json`

4. **Improved Visualization** (utils.py lines 13-58):
   - Now uses **actual tracked entropy values** instead of inferring from branch points
   - Plots entropy as a continuous line over token positions
   - Marks branch points with red stars
   - Shows threshold as horizontal dashed line
   - Proper labels and legend

**Before**: Plot showed branch positions as vertical lines (entropy values unknown)
**After**: Plot shows actual entropy curve with marked branch points

---

### Bug Fixes

**Also Fixed** (from previous session):
1. Missing `traceback` import - Added to all demo scripts
2. Wrong parameter location for `max_paths` - Moved from `generate()` to `__init__()`
3. Plot function now accepts `entropy_data` parameter

---

### New Output Files

When running demos with `--save-plots`, you now get:

```
demo_results/
├── entropy_vs_tokens.png        # ✅ Now with real entropy values!
├── sample_tree.png               # (unchanged)
├── resource_comparison.png       # (unchanged)
├── all_samples.txt               # (unchanged)
└── entropy_data.json             # ✅ NEW! Full entropy history
```

---

### Usage Examples

#### Run with default Qwen model:
```bash
python3 interactive_demo.py
```

#### Override to use different model:
```bash
python3 interactive_demo.py --model gpt2 --prompt "Hello"
```

#### Check entropy data after run:
```bash
cat demo_results/entropy_data.json | jq '.statistics'
```

Output:
```json
{
  "mean_entropy": 0.342,
  "max_entropy": 0.587,
  "min_entropy": 0.123,
  "branch_rate": 0.113
}
```

---

### For Analysis/Experiments

The `entropy_data.json` file is perfect for later analysis:

**Python example**:
```python
import json

# Load entropy data
with open('demo_results/entropy_data.json') as f:
    data = json.load(f)

# Analyze
positions = data['positions']
entropies = data['entropies']
branched = data['branched']

# Find all branch points
branch_positions = [p for p, b in zip(positions, branched) if b]
print(f"Branched at: {branch_positions}")

# Get entropy at branch points
branch_entropies = [e for e, b in zip(entropies, branched) if b]
print(f"Entropy when branching: {branch_entropies}")

# Statistics
print(f"Mean entropy: {data['statistics']['mean_entropy']:.3f}")
print(f"Branch rate: {data['statistics']['branch_rate']:.1%}")
```

---

### Benefits

1. **Better Model Quality**: Qwen produces coherent, high-quality text
2. **Full Observability**: Can now see exact entropy values at each position
3. **Data for Analysis**: JSON file enables downstream analysis and experiments
4. **Debugging**: Can validate EAB branching decisions with actual entropy values
5. **Reproducibility**: Entropy data saved alongside samples for later review

---

### Compatibility

- ✅ Backward compatible: Old demo runs still work
- ✅ Model can be overridden with `--model` flag
- ✅ Entropy plots degrade gracefully if data unavailable
- ✅ All existing functionality preserved

---

**Status**: ✅ Complete and ready to use
**Date**: 2026-01-01
**Files Modified**: 4 (interactive_demo.py, quick_test.py, compare_thresholds.py, utils.py, README.md)
