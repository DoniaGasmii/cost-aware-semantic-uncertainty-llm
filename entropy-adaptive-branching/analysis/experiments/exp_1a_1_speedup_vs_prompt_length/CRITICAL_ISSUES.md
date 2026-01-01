# 🚨 CRITICAL ISSUES FOUND IN INITIAL RESULTS

## Issue 1: EAB IS NOT BRANCHING! ⚠️⚠️⚠️

**Evidence**:
```json
"branch_count": 0,
"branch_positions": [],
"final_path_count": 20
```

**What this means**:
- EAB generated 20 samples with **ZERO branches**
- This defeats the entire purpose of EAB!
- EAB is essentially running naive generation 20 times instead of branching

**Root Cause (TO INVESTIGATE)**:

The problem is likely in `/eab/core.py` in the `generate()` method. Check:

1. **Is `generate()` being called correctly?**
   ```python
   # Current call in run_experiment.py line ~77
   samples = eab.generate(
       prompt=prompt_text,
       max_new_tokens=config['generation']['max_new_tokens'],
       temperature=config['generation']['temperature']
   )
   ```

2. **Does `generate()` actually implement branching logic?**
   - It should start with 1 path
   - Compute entropy at each token
   - Branch when entropy > threshold
   - Return multiple samples from shared computation

3. **Is the entropy threshold too high?**
   - Current: `entropy_threshold = 0.4`
   - If model is always confident (low entropy), it never branches
   - Try lowering to 0.2 or 0.3

**ACTION ITEMS**:
- [ ] Check `/eab/core.py` line ~117 (where EAB is initialized)
- [ ] Verify `generate()` method implements branching
- [ ] Add debug logging to EAB to print entropy values and branching decisions
- [ ] Test with lower entropy threshold

---

## Issue 2: Unfair Token Counting (FIXED ✓)

**Before**:
- Naive: 99 tokens/sample (prompt + generated)
- EAB: 50 tokens/sample (unclear what this counted)

**Fixed in latest commit**:
- Now both track prompt + generated tokens correctly
- Added `num_generated_tokens` field to separate generated from prompt

---

## Issue 3: Inconsistent Timing

**Evidence**:
- Prompt 1: EAB 156s vs Naive 64s → **EAB is 2.4× SLOWER!**
- Prompt 2: EAB 14s vs Naive 231s → **EAB is 16× faster**

**Likely Causes**:
1. **First run overhead**: Model loading, compilation, cache warming
2. **CPU variability**: Other processes interfering
3. **If EAB isn't branching**: Then these numbers are meaningless anyway

**Solution**:
- Ignore first prompt results (warmup)
- Or add explicit warmup run before experiments
- Once EAB branching is fixed, re-run all experiments

---

## Issue 4: GPT-2 Quality

**Observation**: You mentioned responses might be nonsense

**Recommendation**:
- GPT-2 is old (2019) and small (124M parameters)
- Consider using:
  - **GPT-2 Large** (774M) - better quality, still free
  - **GPT-2 XL** (1.5B) - best GPT-2 variant
  - **GPT-Neo 1.3B** - More modern, similar size
  - **GPT-J 6B** - Much better, but needs more memory/time

**For debugging**: Stick with GPT-2 until branching works, then upgrade

---

## Next Steps

### Priority 1: FIX BRANCHING 🔴
This is the most critical issue. Without branching, the entire experiment is invalid.

1. Inspect `/eab/core.py` `generate()` method
2. Add debug prints to see:
   - Entropy values at each position
   - When branching should occur
   - How many paths are active
3. Test with simple prompt to verify branching works

### Priority 2: Verify Token Counting
After fixing branching, ensure:
- `token_steps` accurately reflects computation
- Both methods count the same thing
- Speedup metrics make sense

### Priority 3: Run Clean Experiment
Once branching works:
1. Delete old results
2. Run with debug mode (2 lengths × 2 prompts)
3. Inspect generated texts
4. Verify metrics look reasonable
5. Scale to full experiment

---

## Diagnostic Commands

```bash
# 1. Check EAB implementation
cat /localhome/gasmi/semester_project/cost-aware-semantic-uncertainty-llm/entropy-adaptive-branching/eab/core.py

# 2. Test EAB manually
python3 -c "
from eab import EntropyAdaptiveBranching
eab = EntropyAdaptiveBranching(model_name='gpt2', entropy_threshold=0.3, branch_factor=3, device='cpu')
samples = eab.generate('The capital of France is', max_new_tokens=20)
print(f'Generated {len(samples)} samples')
print(f'Branching: {eab.branch_history if hasattr(eab, \"branch_history\") else \"No branch tracking\"}')
"

# 3. View generated texts
cat results/generated_texts/len50_prompt1_generations.txt
```

---

**Last Updated**: 2025-12-31
**Status**: 🔴 BLOCKING - Cannot proceed until branching is fixed
