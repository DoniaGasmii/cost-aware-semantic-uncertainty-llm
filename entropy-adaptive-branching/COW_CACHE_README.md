# Copy-on-Write (COW) Cache Implementation

## Overview

The COW cache implementation (`eab/cache_cow.py` and `eab/core_cow.py`) solves the memory overhead problem caused by deep copying KV-caches during branching in the EAB algorithm.

## Problem

In the original implementation:
- When branching at high-entropy positions, we create multiple path copies
- Each path requires a **full deep copy** of the KV-cache (500+ MB)
- With 20 paths and frequent branching, this causes significant memory overhead

**Example memory usage (original):**
- Parent cache: 500 MB
- 3 branches × 500 MB each = **1,500 MB total**

## Solution: Copy-on-Write Cache

The COW cache shares the common prefix between branched paths:
- Keep a **reference** to the parent cache (shared prefix)
- Store only the **divergent tokens** in each branch's own cache
- Combine them when needed by the model

**Example memory usage (COW):**
- Parent cache: 500 MB (shared)
- 3 branches × 50 MB divergent each = **650 MB total**
- **Savings: 850 MB (57% reduction)**

## Implementation

### Key Components

#### 1. `CopyOnWriteCache` Class ([cache_cow.py](eab/cache_cow.py))

```python
class CopyOnWriteCache:
    def __init__(self, parent_cache=None, divergence_point=0, device='cuda'):
        self.parent = parent_cache          # Reference to parent (shared)
        self.own_cache = DynamicCache()     # Only divergent part
        self.divergence_point = divergence_point

    def branch(self):
        """Create a new branch that shares this cache as parent"""
        return CopyOnWriteCache(
            parent_cache=self,
            divergence_point=self.get_seq_length(),
            device=self.device
        )

    def _get_combined_layer(self, layer_idx):
        """Combine parent + own cache for model inference"""
        # Get parent and own cache, concatenate them
        ...
```

**Key methods:**
- `branch()` - Create a new branched cache (very cheap, just a reference)
- `update()` - Add new KV states to own cache
- `to_legacy_cache()` - Convert to format the model expects
- `get_memory_usage()` - Track memory usage

#### 2. Modified EAB Implementation ([core_cow.py](eab/core_cow.py))

**Changes from original `core.py`:**

1. **Imports:**
   ```python
   from .cache_cow import CopyOnWriteCache, cow_cache_copy
   ```

2. **Initial cache wrapping:**
   ```python
   # Wrap initial cache from prompt in COW format
   cow_cache = CopyOnWriteCache.from_legacy_cache(
       past_key_values,
       device=self.device
   )
   ```

3. **Cache passing to model:**
   ```python
   # Convert COW cache to format model expects
   if isinstance(cache_to_use, CopyOnWriteCache):
       legacy_cache = cache_to_use.to_legacy_cache()
       cache_to_use = DynamicCache.from_legacy_cache(legacy_cache)
   ```

4. **Branching logic:**
   ```python
   # Use COW branching instead of deep copy
   branch_path.cache = path_cache.branch()  # Fast reference copy!
   ```

#### 3. Modified Path Manager ([path.py](eab/path.py:50-67))

The `GenerationPath.copy()` method now detects and handles COW caches:
```python
if isinstance(self.cache, CopyOnWriteCache):
    copied_cache = self.cache.branch()  # COW branching
else:
    copied_cache = deep_copy_cache(self.cache)  # Regular deep copy
```

## Usage

### Using the COW Implementation

```python
from eab.core_cow import EntropyAdaptiveBranching

# Use exactly like the original implementation
eab = EntropyAdaptiveBranching(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    entropy_threshold=0.055,
    branch_factor=3,
    max_paths=20,
    torch_dtype=torch.float16
)

# Generate samples (same API)
results = eab.generate(
    "What are the benefits of exercise?",
    max_new_tokens=50,
    temperature=0.8
)
```

### Testing and Comparison

Run the test script to compare original vs COW:

```bash
# Quick test (COW only)
python test_cow_implementation.py --quick

# Full comparison (original vs COW)
python test_cow_implementation.py --full
```

## Expected Benefits

### Memory Savings
- **Cache memory**: 60-70% reduction in KV-cache memory during branching
- **Peak memory**: Depends on branching frequency and max_paths
- **Note**: Model weights (6+ GB) dominate total memory, so overall savings are more modest (5-15% of total)

### Generation Quality
- **Identical outputs** (with same seed) to original implementation
- No impact on generation quality or diversity

### Performance
- **Branching**: Much faster (reference copy vs deep copy)
- **Generation**: Slightly slower per token (cache concatenation overhead)
- **Overall**: Similar or better, especially with frequent branching

## Memory Hierarchy

```
Initial Cache (from prompt)
    ├─ Branch 1 Cache (only new tokens)
    │   ├─ Branch 1.1 Cache
    │   └─ Branch 1.2 Cache
    ├─ Branch 2 Cache (only new tokens)
    └─ Branch 3 Cache (only new tokens)
```

Each branch only stores its divergent tokens, all sharing the initial prompt cache.

## Files

- `eab/cache_cow.py` - COW cache implementation
- `eab/core_cow.py` - Modified EAB using COW cache
- `eab/path.py` - Updated to handle COW caches (modified in-place)
- `test_cow_implementation.py` - Testing and comparison script

## Backward Compatibility

The original implementation (`eab/core.py`) remains unchanged and functional. You can use either:
- `from eab.core import EntropyAdaptiveBranching` (original, deep copy)
- `from eab.core_cow import EntropyAdaptiveBranching` (new, COW cache)

## Technical Details

### Cache Update Flow

1. **Model forward pass** with combined cache (parent + own)
2. **Model returns** full updated cache (old + new token)
3. **Extract** just the new token's KV states
4. **Update** COW cache with only the new token
5. **Next iteration** combines parent + accumulated own cache

### Memory Tracking

Get detailed memory statistics:
```python
cache.get_memory_usage()
# Returns:
# {
#     'own_memory_mb': 45.2,      # This branch's memory
#     'parent_memory_mb': 512.1,  # Shared parent memory
#     'total_memory_mb': 557.3,   # Combined total
#     'divergence_point': 43,     # Where this branch diverged
#     'sequence_length': 68       # Total sequence length
# }
```

## Future Improvements

Potential optimizations:
1. Batch cache concatenation across layers
2. Use torch native concatenation operations
3. Implement cache reuse across similar prefixes
4. Integrate with vLLM's Automatic Prefix Caching

## References

- vLLM Automatic Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/
- HuggingFace DynamicCache: https://huggingface.co/docs/transformers/internal/generation_utils
