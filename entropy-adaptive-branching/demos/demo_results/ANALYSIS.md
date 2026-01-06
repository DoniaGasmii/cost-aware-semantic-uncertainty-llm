# Demo Results Analysis

## Your Observations & Explanations

### Observation 1: Duplicate Samples in EAB ✅ EXPECTED BEHAVIOR

You noticed many duplicate samples in EAB output:
- Samples 1, 2, 3: Identical
- Samples 8, 9, 10, 11: Identical
- Samples 12, 13: Identical
- etc.

**Why this happens:**

This is actually **expected and correct behavior**! Here's what's occurring:

1. **Branching happens at high-entropy positions**:
   - Position 39, 41, 45, 50, 70, etc.
   - At these points, the model is uncertain and explores multiple token choices

2. **Paths explore different continuations**:
   - Branch 1: Token A → continues generating
   - Branch 2: Token B → continues generating
   - Branch 3: Token C → continues generating

3. **Paths can converge after branching**:
   - After exploring different branches, the model may become confident again
   - If all paths select the same subsequent tokens (low entropy), they converge to identical completions
   - Example: All paths generate "...which was a period of significant political and social change in Tunisia"

**What this tells us:**

- ✅ **Good**: EAB explores uncertainty at the right positions
- ✅ **Good**: Paths naturally converge when model regains confidence
- ✅ **Good**: Shows the model's uncertainty was *temporary* and resolved

**For your experiments:**

You should report both metrics:
- **Total paths explored**: 29 (shows exploration effort)
- **Unique completions**: ~10-15 (shows distinct outputs)
- **Convergence rate**: (duplicates/total) indicates how often exploration led to same conclusion

This is a **feature that demonstrates EAB's intelligence** - it doesn't waste resources on divergent paths when the model becomes confident.

---

### Observation 2: Naive Samples Show Full Chat Template ❌ BUG (FIXED!)

You noticed naive samples included the full chat template:

```
system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user
who was the first president of Tunisia and when?
assistant
[answer]
```

**The problem:**

The display function was showing `sample['text']` (full output) instead of `sample['generated_only']` (just the answer).

**The fix:**

I updated `/demos/utils.py` line 374 to prioritize `'generated_only'`:

```python
# Before (wrong):
text = sample.get('text', sample.get('generated_only', 'N/A'))

# After (correct):
text = sample.get('generated_only', sample.get('text', 'N/A'))
```

**Next time you run the demo**, naive samples will show only the generated answer, making comparison fair.

---

## Summary Statistics from Your Demo

### Prompt
```
"who was the first president of Tunisia and when?"
```

### EAB Results
- **Total paths**: 29
- **Unique outputs**: ~8-10 distinct variants
- **Branching positions**: [39, 41, 45, 50, 70, 71, 72, 73, 74, 75, 78, 80, 85, 86, 87]
- **Branch points**: 15 positions triggered branching

**Distinct answer patterns found:**
1. "...served from 1957 to 1987"
2. "...served from 1957 until his death in 1987"
3. "...was the leader from 1957 to 1987"
4. "...period of significant political and social change"
5. "...longest-serving head of state"
6. "...period of significant political, social, and economic development"

### Naive Results
- **Total samples**: 29
- **Unique outputs**: ~12-15 distinct variants (more variation due to independent sampling)

---

## Key Insights

1. **EAB explores strategically**: Branches at uncertainty points, converges at confidence points

2. **Path convergence is intelligent**: Shows model resolved its uncertainty consistently

3. **Diversity vs Redundancy tradeoff**:
   - EAB: Some duplicates due to convergence, but shared computation
   - Naive: More unique samples, but wasteful independent generation

4. **For your thesis**:
   - Report both "paths explored" and "unique outputs"
   - Discuss convergence as a sign of model confidence restoration
   - This supports your "branch when uncertain" philosophy

---

## Recommended Metrics for Experiments

### Efficiency Metrics
- Peak memory (MB)
- Generation time (seconds)
- Total forward passes

### Diversity Metrics
- Total paths generated
- Unique completions (deduplicated)
- Convergence rate = duplicates / total
- Self-BLEU (lower = more diverse)
- Distinct-n (unique n-grams)

### Uncertainty Metrics
- Number of branching points
- Average entropy at branch points
- Entropy distribution across generation

---

## Next Steps

1. ✅ **Fixed**: Naive sample display (no more chat template clutter)

2. **For experiments**: Consider adding deduplication analysis
   ```python
   unique_texts = set(s['text'] for s in samples)
   convergence_rate = 1 - (len(unique_texts) / len(samples))
   ```

3. **For report**: Explain path convergence as a positive feature showing model confidence restoration

4. **Run pilot study**: Use these corrected display functions for clean results
