"""
Tests for KV-cache management module.
"""

import pytest
import torch
from eab.cache import (
    deep_copy_cache,
    shallow_copy_cache,
    get_cache_size,
    validate_cache,
    truncate_cache,
    CacheManager
)


def create_dummy_cache(num_layers=2, batch_size=1, num_heads=4, seq_len=10, head_dim=16):
    """Create a dummy KV-cache for testing."""
    cache = tuple(
        tuple(
            torch.randn(batch_size, num_heads, seq_len, head_dim)
            for _ in range(2)  # key and value
        )
        for _ in range(num_layers)
    )
    return cache


class TestCacheOperations:
    """Test basic cache operations."""
    
    def test_deep_copy_cache(self):
        """Test deep copying of cache."""
        original = create_dummy_cache(num_layers=2, seq_len=5)
        copied = deep_copy_cache(original)
        
        # Should have same structure
        assert len(copied) == len(original)
        
        # Should have same values
        for orig_layer, copy_layer in zip(original, copied):
            for orig_kv, copy_kv in zip(orig_layer, copy_layer):
                assert torch.allclose(orig_kv, copy_kv)
        
        # Modifications shouldn't affect original
        copied[0][0][0, 0, 0, 0] = 999.0
        assert original[0][0][0, 0, 0, 0] != 999.0
    
    def test_deep_copy_none(self):
        """Test deep copying None."""
        assert deep_copy_cache(None) is None
    
    def test_shallow_copy_cache(self):
        """Test shallow copying of cache."""
        original = create_dummy_cache(num_layers=2, seq_len=5)
        copied = shallow_copy_cache(original)
        
        # Should have same structure
        assert len(copied) == len(original)
        
        # Modifications SHOULD affect original (shared tensors)
        copied[0][0][0, 0, 0, 0] = 999.0
        assert original[0][0][0, 0, 0, 0] == 999.0
    
    def test_get_cache_size(self):
        """Test getting cache size information."""
        cache = create_dummy_cache(
            num_layers=3,
            batch_size=2,
            num_heads=8,
            seq_len=20,
            head_dim=32
        )
        
        size_info = get_cache_size(cache)
        
        assert size_info['num_layers'] == 3
        assert size_info['batch_size'] == 2
        assert size_info['num_heads'] == 8
        assert size_info['sequence_length'] == 20
        assert size_info['head_dim'] == 32
        assert size_info['memory_mb'] > 0
    
    def test_get_cache_size_none(self):
        """Test getting size of None cache."""
        size_info = get_cache_size(None)
        
        assert size_info['num_layers'] == 0
        assert size_info['memory_mb'] == 0.0
    
    def test_validate_cache_valid(self):
        """Test validation of valid cache."""
        cache = create_dummy_cache(num_layers=2)
        
        assert validate_cache(cache) is True
    
    def test_validate_cache_none(self):
        """Test validation of None cache."""
        assert validate_cache(None) is True
    
    def test_validate_cache_invalid_structure(self):
        """Test validation of invalid cache structure."""
        # Wrong number of key-value pairs
        invalid_cache = (
            (torch.randn(1, 4, 10, 16),),  # Only one tensor instead of two
        )
        
        assert validate_cache(invalid_cache) is False
    
    def test_validate_cache_wrong_dimensions(self):
        """Test validation of cache with wrong tensor dimensions."""
        # 3D tensors instead of 4D
        invalid_cache = (
            (
                torch.randn(4, 10, 16),  # Missing batch dimension
                torch.randn(4, 10, 16)
            ),
        )
        
        assert validate_cache(invalid_cache) is False
    
    def test_truncate_cache(self):
        """Test cache truncation."""
        cache = create_dummy_cache(num_layers=2, seq_len=20)
        
        truncated = truncate_cache(cache, max_length=10)
        
        # Check new sequence length
        assert truncated[0][0].shape[2] == 10
        
        # Original should be unchanged
        assert cache[0][0].shape[2] == 20
    
    def test_truncate_cache_shorter_than_max(self):
        """Test truncating cache that's already shorter than max."""
        cache = create_dummy_cache(num_layers=2, seq_len=5)
        
        truncated = truncate_cache(cache, max_length=10)
        
        # Should remain same size
        assert truncated[0][0].shape[2] == 5


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = CacheManager(max_cache_memory_mb=1000.0)
        
        assert manager.max_cache_memory_mb == 1000.0
        assert manager.cache_stats['num_copies'] == 0
    
    def test_copy_for_branching(self):
        """Test creating copies for branching."""
        manager = CacheManager()
        cache = create_dummy_cache(num_layers=2, seq_len=10)
        
        copies = manager.copy_for_branching(cache, num_branches=3)
        
        assert len(copies) == 3
        assert manager.cache_stats['num_copies'] == 3
        
        # Each should be independent
        copies[0][0][0][0, 0, 0, 0] = 999.0
        assert copies[1][0][0][0, 0, 0, 0] != 999.0
    
    def test_check_memory_limit(self):
        """Test memory limit checking."""
        manager = CacheManager(max_cache_memory_mb=1.0)  # Very small limit
        
        # Small cache should fit
        small_cache = create_dummy_cache(num_layers=1, seq_len=5, head_dim=8)
        assert manager.check_memory_limit(2, small_cache) is True
        
        # Large cache should not fit
        large_cache = create_dummy_cache(num_layers=10, seq_len=1000, head_dim=64)
        assert manager.check_memory_limit(10, large_cache) is False
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        manager = CacheManager()
        cache = create_dummy_cache(num_layers=2)
        
        manager.copy_for_branching(cache, num_branches=3)
        
        assert manager.cache_stats['num_copies'] == 3
        
        manager.reset_statistics()
        
        assert manager.cache_stats['num_copies'] == 0
        assert manager.cache_stats['total_memory_mb'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])