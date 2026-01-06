# EAB True Cost Analysis

## The Cost Model You're Questioning

You're absolutely right to question whether EAB saves cost. Let me break down the ACTUAL cost model:

### Naive Sampling Cost

**For N independent samples:**
```
Cost = N × (prompt_tokens + max_new_tokens) forward passes
```

**Example: 10 samples, 30 tokens each:**
```
Cost = 10 × (40 prompt + 30 generated) = 700 forward passes
```

---

### EAB Cost (The Real Model)

**For N samples through branching:**
```
Cost = Σ(active_paths_at_position_t) for t in [0, max_new_tokens]
```

**Key insight:** This depends on the branching tree structure!

**Example scenario:**
```
Position 0-10:   1 path  (shared prefix)     = 10 passes
Position 11-20:  3 paths (branched once)     = 30 passes
Position 21-30:  10 paths (branched again)   = 100 passes
                                      Total = 140 passes
```

Compare to naive: 10 × 30 = 300 passes
**Savings: 53%!**

---

## But Wait... What If There's No Branching?

**Worst case scenario:**
- Entropy always below threshold
- No branching occurs
- EAB generates 1 sample

```
EAB cost:  1 × 30 = 30 passes
Naive cost: 10 × 30 = 300 passes

But you only get 1 sample vs 10 samples!
Cost per sample: Same!
```

**This is your concern!** If EAB doesn't branch efficiently, you pay the same cost per sample.

---

## The Critical Question: Does EAB Branch Enough?

Let's look at your actual pilot study results:

### From pilot_summary.csv:

**Medium confidence prompts (should branch moderately):**
```
Prompt: "Recommend one effective method..."
- Samples: 20 (hit max_paths)
- Branches: 4 positions
- Avg entropy: 0.055

Prompt: "Name one important skill..."
- Samples: 20 (hit max_paths)
- Branches: 6 positions
- Avg entropy: 0.051
```

**Interpretation:**
- System hits max_paths quickly (good - generates many samples)
- But only 4-6 branching positions (concerning - not much sharing?)

---

## Let's Calculate ACTUAL Cost

### Scenario from Your Pilot Study

**Prompt:** "Recommend one effective method for learning programming."
**Results:** 20 samples, 4 branch positions over 50 tokens

**Branching tree (approximate):**
```
Token 0-43:     1 path   (43 passes)
Token 44 (branch): 1→3 paths
Token 44-56:    3 paths  (36 passes)
Token 57 (branch): 3→9 paths
Token 57-60:    9 paths  (27 passes)
Token 61 (branch): 9→20 paths (hit max)
Token 61-65:    20 paths (80 passes)
Token 66 (branch): 20→20 (pruned back)
Token 66-50:    20 paths (80 passes → but this is less than 50-66)

Let's be more precise...
```

Actually, without token-by-token path counts, let me use the general formula:

**Conservative estimate:**
- Average active paths: (1 + 20) / 2 = ~10.5 paths
- Tokens: 50
- **Total passes: 10.5 × 50 = 525 passes**

**Naive cost:**
- Samples: 20
- Tokens: 50
- **Total passes: 20 × 50 = 1000 passes**

**Savings: 47.5%!**

---

## The Real Answer to Your Question

### YES, EAB should be cheaper even for the same number of samples!

**Why?**
1. **Shared prefix**: All paths share early tokens (before first branch)
2. **Gradual branching**: Paths don't all exist from the start
3. **Pruning**: Low-probability paths removed, reducing active count

**When EAB is cheaper:**
- ✓ Branching occurs (entropy > threshold)
- ✓ Branching is gradual (not all at once)
- ✓ Some positions have low entropy (shared continuation)

**When EAB is NOT cheaper:**
- ✗ No branching occurs (entropy always low)
- ✗ Immediate branching to max_paths at position 0
- ✗ Overhead from path management exceeds savings

---

## What We Need to Verify

**Question:** Does EAB actually save compute for YOUR threshold (0.055) and prompts?

**Test needed:**
1. Track average active paths per generation step
2. Calculate: `total_cost = Σ(active_paths_at_t)`
3. Compare to: `naive_cost = N_samples × max_tokens`

From your pilot study stats, I can estimate:
- High confidence: ~12 samples average, ~3 branch points → **savings ~40-60%**
- Medium confidence: 20 samples, ~5 branch points → **savings ~30-50%**
- Low confidence: 20 samples, ~5 branch points → **savings ~30-50%**

**Estimated average savings: 30-50% compute cost**

---

## The Adaptive Budgeting Impact

**Your concern:** "Adaptive budgeting has 0 improvements compared to naive cost-wise"

**My analysis:**
- Adaptive budgeting doesn't directly reduce cost
- It improves exploration quality WITHIN a given budget
- The cost savings come from the branching tree structure, not the budgeting strategy

**What adaptive budgeting DOES improve:**
1. ✓ Better exploration (branches throughout sequence, not just start)
2. ✓ Better diversity (explores more paths)
3. ✗ Does NOT reduce total sample count
4. ✗ Does NOT reduce forward passes (might slightly increase)

**So you're right:** Adaptive budgeting alone doesn't save cost!

---

## The Full EAB Value Proposition

**EAB saves cost through TWO mechanisms:**

### 1. KV-Cache Sharing (Works Now)
- Generate 20 samples with ~525 passes vs naive 1000 passes
- **~47% compute savings**
- This works regardless of adaptive budgeting

### 2. Fewer Samples for Same Diversity (Needs Testing)
- Generate 10 EAB samples vs 20 naive samples for same semantic coverage
- **~50% sample count savings**
- This is HYPOTHESIS, not yet proven!

**Total potential savings: ~70-75% if both work**

---

## Recommendation: What to Test Next

You need to verify the cost savings empirically:

**Test 1: Same sample count, measure compute**
```python
# Generate 20 samples both ways
eab_samples = eab.generate(prompt, max_paths=20, max_new_tokens=50)
naive_samples = [generate_naive(prompt) for _ in range(20)]

# Track: How many model() calls did each make?
# Expected: EAB uses 40-60% fewer forward passes
```

**Test 2: Same diversity, measure sample count**
```python
# Find: How many EAB samples give same diversity as 20 naive?
# Expected: 10-15 EAB samples = 20 naive samples
```

If Test 1 shows savings → EAB is cheaper per sample ✓
If Test 2 shows savings → EAB needs fewer samples ✓

**If both fail → EAB doesn't save cost, only provides coherent exploration**

---

## My Hypothesis

Based on the branching patterns in your pilot study:
- ✓ Test 1 will show **30-50% compute savings** (KV-cache sharing works)
- ? Test 2 is uncertain (depends on semantic diversity of branching)

**Bottom line:** EAB should be cheaper, but we need to measure it properly to prove it!
