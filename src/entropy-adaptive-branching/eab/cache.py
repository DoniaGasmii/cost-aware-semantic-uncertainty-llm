"""
KV-cache management for efficient multi-path generation.

Handles copying, merging, and updating of transformer KV-caches
across different generation paths.
"""

import torch
from typing import Tuple, Optional, List
import copy


def deep_copy_cache(cache: Optional[Tuple]) -> Optional[Tuple]:
    """
    Create a deep copy of a KV-cache for branching.
    
    The cache structure from HuggingFace transformers is:
    Tuple of tuples, where each inner tuple represents a layer:
    (
        (key_layer_0, value_layer_0),
        (key_layer_1, value_layer_1),
        ...
    )
    
    Each key/value is a tensor of shape:
    [batch_size, num_heads, sequence_length, head_dim]
    
    Args:
        cache: KV-cache tuple from transformers model
    
    Returns:
        Deep copy of the cache, or None if input is None
    
    Examples:
        >>> outputs = model(input_ids, use_cache=True)
        >>> original_cache = outputs.past_key_values
        >>> copied_cache = deep_copy_cache(original_cache)
    """
    if cache is None:
        return None
    
    # Deep copy each layer's key-value pair
    copied_cache = tuple(
        tuple(kv.clone() for kv in layer_cache)
        for layer_cache in cache
    )
    
    return copied_cache


def shallow_copy_cache(cache: Optional[Tuple]) -> Optional[Tuple]:
    """
    Create a shallow copy of a KV-cache (shares tensor data).
    
    Useful when you want to modify the cache structure but not the tensors.
    WARNING: Changes to tensors will affect both caches!
    
    Args:
        cache: KV-cache tuple from transformers model
    
    Returns:
        Shallow copy of the cache
    """
    if cache is None:
        return None
    
    return tuple(tuple(kv for kv in layer_cache) for layer_cache in cache)


def concatenate_caches(cache1: Tuple, cache2: Tuple, dim: int = 2) -> Tuple:
    """
    Concatenate two KV-caches along the sequence dimension.
    
    Useful for combining caches from different generation steps.
    
    Args:
        cache1: First KV-cache
        cache2: Second KV-cache
        dim: Dimension to concatenate along (default: 2 for sequence length)
    
    Returns:
        Concatenated cache
    """
    if cache1 is None:
        return cache2
    if cache2 is None:
        return cache1
    
    concatenated = tuple(
        tuple(
            torch.cat([kv1, kv2], dim=dim)
            for kv1, kv2 in zip(layer1, layer2)
        )
        for layer1, layer2 in zip(cache1, cache2)
    )
    
    return concatenated


def truncate_cache(cache: Tuple, max_length: int) -> Tuple:
    """
    Truncate a KV-cache to a maximum sequence length.
    
    Useful for managing memory when sequences get very long.
    
    Args:
        cache: KV-cache to truncate
        max_length: Maximum sequence length to keep
    
    Returns:
        Truncated cache
    """
    if cache is None:
        return None
    
    truncated = tuple(
        tuple(
            kv[:, :, -max_length:, :] if kv.size(2) > max_length else kv
            for kv in layer_cache
        )
        for layer_cache in cache
    )
    
    return truncated


def get_cache_size(cache: Optional[Tuple]) -> dict:
    """
    Get size information about a KV-cache.
    
    Args:
        cache: KV-cache to analyze
    
    Returns:
        Dictionary with cache dimensions and memory usage
    
    Examples:
        >>> size_info = get_cache_size(cache)
        >>> print(f"Sequence length: {size_info['sequence_length']}")
        >>> print(f"Memory (MB): {size_info['memory_mb']:.2f}")
    """
    if cache is None:
        return {
            'num_layers': 0,
            'sequence_length': 0,
            'batch_size': 0,
            'num_heads': 0,
            'head_dim': 0,
            'memory_mb': 0.0
        }
    
    # Get dimensions from first layer's first key tensor
    first_key = cache[0][0]
    batch_size, num_heads, seq_len, head_dim = first_key.shape
    num_layers = len(cache)
    
    # Calculate memory usage
    total_elements = 0
    element_size = 0
    
    for layer_cache in cache:
        for kv_tensor in layer_cache:
            total_elements += kv_tensor.numel()
            element_size = kv_tensor.element_size()
    
    memory_bytes = total_elements * element_size
    memory_mb = memory_bytes / (1024 * 1024)
    
    return {
        'num_layers': num_layers,
        'sequence_length': seq_len,
        'batch_size': batch_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'total_elements': total_elements,
        'memory_mb': memory_mb
    }


def merge_caches(caches: List[Tuple]) -> Tuple:
    """
    Merge multiple KV-caches by stacking along batch dimension.
    
    Useful for batch processing multiple paths simultaneously.
    
    Args:
        caches: List of KV-caches to merge
    
    Returns:
        Merged cache with batch_size = sum of input batch sizes
    """
    if not caches:
        return None
    
    if len(caches) == 1:
        return caches[0]
    
    # Stack along batch dimension (dim=0)
    merged = tuple(
        tuple(
            torch.cat([cache[layer_idx][kv_idx] for cache in caches], dim=0)
            for kv_idx in range(2)  # key and value
        )
        for layer_idx in range(len(caches[0]))
    )
    
    return merged


def split_cache(cache: Tuple, batch_sizes: List[int]) -> List[Tuple]:
    """
    Split a batched KV-cache into individual caches.
    
    Inverse operation of merge_caches.
    
    Args:
        cache: Batched KV-cache
        batch_sizes: List of batch sizes for each split
    
    Returns:
        List of individual KV-caches
    """
    if cache is None:
        return []
    
    splits = []
    start_idx = 0
    
    for batch_size in batch_sizes:
        split_cache = tuple(
            tuple(
                kv[start_idx:start_idx + batch_size]
                for kv in layer_cache
            )
            for layer_cache in cache
        )
        splits.append(split_cache)
        start_idx += batch_size
    
    return splits


class CacheManager:
    """
    Manages KV-caches for multiple generation paths.
    
    Provides high-level interface for cache operations during
    entropy-adaptive branching.
    """
    
    def __init__(self, max_cache_memory_mb: float = 1000.0):
        """
        Initialize cache manager.
        
        Args:
            max_cache_memory_mb: Maximum memory (in MB) to use for caches
        """
        self.max_cache_memory_mb = max_cache_memory_mb
        self.cache_stats = {
            'num_copies': 0,
            'num_merges': 0,
            'total_memory_mb': 0.0,
            'peak_memory_mb': 0.0
        }
    
    def copy_for_branching(self, cache: Tuple, num_branches: int) -> List[Tuple]:
        """
        Create multiple copies of a cache for branching.
        
        Args:
            cache: Original cache to copy
            num_branches: Number of copies to create
        
        Returns:
            List of cache copies
        """
        copies = [deep_copy_cache(cache) for _ in range(num_branches)]
        
        # Update statistics
        self.cache_stats['num_copies'] += num_branches
        cache_size = get_cache_size(cache)
        self.cache_stats['total_memory_mb'] += cache_size['memory_mb'] * num_branches
        self.cache_stats['peak_memory_mb'] = max(
            self.cache_stats['peak_memory_mb'],
            self.cache_stats['total_memory_mb']
        )
        
        return copies
    
    def check_memory_limit(self, num_paths: int, cache: Tuple) -> bool:
        """
        Check if creating num_paths with given cache would exceed memory limit.
        
        Args:
            num_paths: Number of paths to check
            cache: Sample cache to estimate size
        
        Returns:
            True if within limit, False otherwise
        """
        cache_size = get_cache_size(cache)
        estimated_memory = cache_size['memory_mb'] * num_paths
        return estimated_memory <= self.max_cache_memory_mb
    
    def get_statistics(self) -> dict:
        """Get cache usage statistics."""
        return self.cache_stats.copy()
    
    def reset_statistics(self):
        """Reset cache statistics."""
        self.cache_stats = {
            'num_copies': 0,
            'num_merges': 0,
            'total_memory_mb': 0.0,
            'peak_memory_mb': 0.0
        }


def optimize_cache_memory(cache: Tuple, dtype: torch.dtype = torch.float16) -> Tuple:
    """
    Optimize cache memory usage by converting to lower precision.
    
    Args:
        cache: Original cache
        dtype: Target data type (default: float16)
    
    Returns:
        Cache with optimized memory usage
    
    Warning:
        Converting to lower precision may affect generation quality slightly.
    """
    if cache is None:
        return None
    
    optimized = tuple(
        tuple(
            kv.to(dtype=dtype)
            for kv in layer_cache
        )
        for layer_cache in cache
    )
    
    return optimized


def validate_cache(cache: Optional[Tuple]) -> bool:
    """
    Validate that a cache has the correct structure.
    
    Args:
        cache: Cache to validate
    
    Returns:
        True if valid, False otherwise
    """
    if cache is None:
        return True
    
    try:
        # Check it's a tuple of tuples
        if not isinstance(cache, tuple):
            return False
        
        for layer_cache in cache:
            if not isinstance(layer_cache, tuple):
                return False
            
            # Each layer should have 2 tensors (key and value)
            if len(layer_cache) != 2:
                return False
            
            # Check tensors have 4 dimensions [batch, heads, seq, head_dim]
            for kv in layer_cache:
                if not isinstance(kv, torch.Tensor):
                    return False
                if kv.dim() != 4:
                    return False
        
        return True
    
    except Exception:
        return False