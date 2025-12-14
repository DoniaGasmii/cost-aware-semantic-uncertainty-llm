"""
Tests for entropy computation module.
"""

import pytest
import torch
import math
from eab.entropy import (
    compute_entropy,
    normalize_entropy,
    compute_normalized_entropy,
    should_branch,
    get_top_k_tokens,
    EntropyTracker
)


class TestEntropyComputation:
    """Test entropy computation functions."""
    
    def test_compute_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        # Uniform logits
        logits = torch.ones(100)
        entropy = compute_entropy(logits)
        
        # Uniform distribution has maximum entropy
        expected = math.log(100)
        assert abs(entropy - expected) < 0.01
    
    def test_compute_entropy_peaked(self):
        """Test entropy with peaked distribution."""
        # One very large logit
        logits = torch.zeros(100)
        logits[0] = 100.0
        entropy = compute_entropy(logits)
        
        # Peaked distribution has low entropy
        assert entropy < 0.1
    
    def test_compute_entropy_temperature(self):
        """Test temperature effect on entropy."""
        logits = torch.randn(100)
        
        entropy_low_temp = compute_entropy(logits, temperature=0.5)
        entropy_high_temp = compute_entropy(logits, temperature=2.0)
        
        # Higher temperature → higher entropy
        assert entropy_high_temp > entropy_low_temp
    
    def test_normalize_entropy(self):
        """Test entropy normalization."""
        vocab_size = 50000
        max_entropy = math.log(vocab_size)
        
        # Test various entropy values
        assert normalize_entropy(0, vocab_size) == 0.0
        assert normalize_entropy(max_entropy, vocab_size) == 1.0
        assert 0 < normalize_entropy(max_entropy / 2, vocab_size) < 1
    
    def test_compute_normalized_entropy(self):
        """Test combined computation and normalization."""
        logits = torch.randn(50000)
        vocab_size = 50000
        
        normalized = compute_normalized_entropy(logits, vocab_size)
        
        assert 0 <= normalized <= 1
    
    def test_should_branch(self):
        """Test branching decision."""
        vocab_size = 100
        
        # High entropy logits
        high_entropy_logits = torch.ones(vocab_size)
        assert should_branch(high_entropy_logits, vocab_size, threshold=0.4)
        
        # Low entropy logits
        low_entropy_logits = torch.zeros(vocab_size)
        low_entropy_logits[0] = 100.0
        assert not should_branch(low_entropy_logits, vocab_size, threshold=0.4)
    
    def test_get_top_k_tokens(self):
        """Test top-k token extraction."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5, 2.5])
        
        token_ids, probs, log_probs = get_top_k_tokens(logits, k=3)
        
        assert len(token_ids) == 3
        assert len(probs) == 3
        assert len(log_probs) == 3
        assert token_ids[0] == 1  # Index of largest logit
        assert probs.sum() < 1.0 or abs(probs.sum() - 1.0) < 0.01  # Subset of distribution


class TestEntropyTracker:
    """Test entropy tracking functionality."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = EntropyTracker()
        
        assert len(tracker.entropy_history) == 0
        assert len(tracker.position_history) == 0
        assert len(tracker.branching_decisions) == 0
    
    def test_tracker_record(self):
        """Test recording entropy values."""
        tracker = EntropyTracker()
        
        tracker.record(0, 0.5, False)
        tracker.record(1, 0.7, True)
        tracker.record(2, 0.3, False)
        
        assert len(tracker.entropy_history) == 3
        assert tracker.entropy_history[1] == 0.7
        assert tracker.branching_decisions[1] == True
    
    def test_tracker_statistics(self):
        """Test statistics computation."""
        tracker = EntropyTracker()
        
        tracker.record(0, 0.3, False)
        tracker.record(1, 0.5, True)
        tracker.record(2, 0.7, True)
        tracker.record(3, 0.4, False)
        
        stats = tracker.get_statistics()
        
        assert 'mean_entropy' in stats
        assert 'num_branches' in stats
        assert 'branch_rate' in stats
        
        assert abs(stats['mean_entropy'] - 0.475) < 0.01
        assert stats['num_branches'] == 2
        assert abs(stats['branch_rate'] - 0.5) < 0.01
    
    def test_tracker_reset(self):
        """Test tracker reset."""
        tracker = EntropyTracker()
        
        tracker.record(0, 0.5, False)
        tracker.record(1, 0.7, True)
        
        tracker.reset()
        
        assert len(tracker.entropy_history) == 0
        assert len(tracker.position_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])