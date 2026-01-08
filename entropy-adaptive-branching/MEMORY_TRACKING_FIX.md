# Memory Tracking Fix for Experiment 1.A

## Problem Identified

The experiment results showed **incorrect memory measurements**:
- **Naive memory: 0.00 MB** (incorrect!)
- **EAB memory: 8-10 MB** (correct)
- **Speedup memory: 0.00×** (division by zero)

### Root Cause

The `MetricsTracker` class in `analysis/utils/metrics.py` was using **`torch.cuda.memory_allocated()`** which returns the **current** allocated memory, not the peak.

**What happened during naive generation:**
```python
for i in range(num_samples):
    output = model.generate(...)  # Allocates KV cache temporarily
    # KV cache is freed after generate() returns
    tracker.update_memory()  # Called AFTER memory is freed
    # Result: current_memory == start_memory → peak = 0 MB
```

The KV cache was allocated and freed **within** each `model.generate()` call, so by the time we checked memory, it was already freed.

## The Fix

Changed to use **`torch.cuda.max_memory_allocated()`** which tracks the true peak memory since the last reset, even if that memory has been freed.

### Changes Made

**File:** `analysis/utils/metrics.py`

1. **In `start()` method** (line 58-70):
   ```python
   # Reset peak memory stats and record baseline
   if self.device == "cuda" and torch.cuda.is_available():
       torch.cuda.reset_peak_memory_stats()  # ← NEW
       self.start_memory = torch.cuda.memory_allocated() / 1024 / 1024
   ```

2. **In `stop()` method** (line 97-128):
   ```python
   # Calculate peak memory overhead (generation only, excluding model weights)
   if self.device == "cuda" and torch.cuda.is_available():
       # Use max_memory_allocated to capture true peak (even if memory was freed)
       peak_total = torch.cuda.max_memory_allocated() / 1024 / 1024  # ← NEW
       memory_peak_mb = peak_total - self.start_memory
   ```

3. **In `update_memory()` method** (line 72-78):
   ```python
   # For CUDA, we use max_memory_allocated() in stop()
   # For CPU, we need to track manually
   if self.device != "cuda":
       current_memory = self._get_memory_usage()
       self.peak_memory = max(self.peak_memory, current_memory)
   ```

## Impact

This fix ensures that:
- ✅ **Naive memory** is correctly measured (should be 5-20 MB for sequential generation)
- ✅ **EAB memory** continues to be correctly measured (50-100 MB for parallel paths)
- ✅ **Memory speedup** is correctly calculated (should be < 1.0, meaning EAB uses MORE memory)
- ✅ Memory measurements exclude model weights (only generation overhead)

## Expected Results After Fix

With correct memory tracking:
- **Naive**: 5-20 MB (1 sequential path)
- **EAB**: 50-100 MB (20 parallel paths)
- **Memory ratio**: 0.1-0.4× (EAB uses MORE memory than naive)
- **Time speedup**: 1.5-2.0× (EAB is FASTER than naive)

**This is expected!** EAB trades memory for speed:
- Uses more memory (maintains multiple paths)
- But saves time (parallelizes generation, reduces redundant token processing)

## COW Tracking Support

The experiments now also support tracking **both COW and Original** implementations:

**File:** `experiments/exp_1a_1_speedup_vs_prompt_length/config.yaml`

```yaml
eab:
  implementation: "cow"  # "cow", "original", or "both"
```

- **"cow"** (default): Uses Copy-on-Write cache (memory efficient)
- **"original"**: Uses deep copy cache (baseline)
- **"both"**: Runs with BOTH implementations for direct comparison

See [COW_EXPERIMENT_GUIDE.md](analysis/experiments/COW_EXPERIMENT_GUIDE.md) for details.

## Next Steps

1. **Test the fix:**
   ```bash
   cd analysis
   python test_memory_fix.py
   ```

2. **Re-run experiments** with corrected memory tracking:
   ```bash
   cd experiments/exp_1a_1_speedup_vs_prompt_length

   # Option 1: Run with COW only (default)
   python run_experiment.py

   # Option 2: Compare COW vs Original (edit config.yaml: implementation: "both")
   python run_experiment.py
   ```

3. **Verify results** show non-zero naive memory and proper COW vs Original comparison

## Same Issue in Interactive Demo?

No - the interactive demo ([demos/interactive_demo.py](../demos/interactive_demo.py)) was already using the correct approach:
```python
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()
# ... generation ...
peak_total = torch.cuda.max_memory_allocated()
peak_overhead = peak_total - mem_before
```

This fix brings the experiment tracking in line with the demo implementation.
