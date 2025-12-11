"""
Tests for generation path tracking module.
"""

import pytest
import torch
from eab.path import GenerationPath, PathManager


class TestGenerationPath:
    """Test GenerationPath functionality."""
    
    def test_path_initialization(self):
        """Test path initialization."""
        path = GenerationPath(
            tokens=[1, 2, 3],
            log_prob=-2.5,
            path_id=0
        )
        
        assert path.tokens == [1, 2, 3]
        assert path.log_prob == -2.5
        assert path.length == 3
        assert len(path.branch_points) == 0
    
    def test_path_copy(self):
        """Test path copying."""
        original = GenerationPath(
            tokens=[1, 2, 3],
            log_prob=-2.5,
            branch_points=[1, 3],
            path_id=0
        )
        
        copied = original.copy(new_path_id=1)
        
        assert copied.tokens == original.tokens
        assert copied.log_prob == original.log_prob
        assert copied.branch_points == original.branch_points
        assert copied.parent_id == 0
        assert copied.path_id == 1
        
        # Modifications to copy shouldn't affect original
        copied.tokens.append(4)
        assert len(original.tokens) == 3
        assert len(copied.tokens) == 4
    
    def test_add_token(self):
        """Test adding tokens."""
        path = GenerationPath(tokens=[], log_prob=0.0)
        
        path.add_token(5, -0.5)
        path.add_token(10, -1.2)
        
        assert path.tokens == [5, 10]
        assert abs(path.log_prob - (-1.7)) < 0.01
        assert path.length == 2
    
    def test_probability_conversion(self):
        """Test log probability to probability conversion."""
        path = GenerationPath(tokens=[1], log_prob=-2.0)
        
        prob = path.probability
        expected = torch.exp(torch.tensor(-2.0)).item()
        
        assert abs(prob - expected) < 0.01
    
    def test_mark_branch(self):
        """Test branch marking."""
        path = GenerationPath(tokens=[], log_prob=0.0)
        
        path.mark_branch(5)
        path.mark_branch(10)
        path.mark_branch(5)  # Duplicate
        
        # Duplicate shouldn't be added again
        assert path.branch_points == [5, 10]


class TestPathManager:
    """Test PathManager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = PathManager(max_paths=10)
        
        assert manager.max_paths == 10
        assert len(manager.paths) == 0
        assert len(manager.completed_paths) == 0
    
    def test_create_initial_path(self):
        """Test initial path creation."""
        manager = PathManager(max_paths=10)
        
        path = manager.create_initial_path()
        
        assert len(manager.paths) == 1
        assert path.path_id == 0
        assert path.length == 0
    
    def test_branch_path(self):
        """Test path branching."""
        manager = PathManager(max_paths=20)
        
        original = GenerationPath(
            tokens=[1, 2, 3],
            log_prob=-1.5,
            path_id=0
        )
        
        branches = manager.branch_path(original, branch_factor=3, position=5)
        
        assert len(branches) == 3
        
        for branch in branches:
            assert branch.tokens == [1, 2, 3]
            assert branch.log_prob == -1.5
            assert 5 in branch.branch_points
            assert branch.parent_id == 0
    
    def test_prune_paths(self):
        """Test path pruning."""
        manager = PathManager(max_paths=3)
        
        # Create paths with different probabilities
        for i in range(5):
            path = GenerationPath(
                tokens=[i],
                log_prob=-i * 2.0,  # Decreasing probability
                path_id=i
            )
            manager.paths.append(path)
        
        # Should have 5 paths
        assert len(manager.paths) == 5
        
        # Prune to max_paths
        manager.prune_paths()
        
        # Should keep only top 3
        assert len(manager.paths) == 3
        
        # Check they're the highest probability ones
        assert all(p.log_prob >= -6.0 for p in manager.paths)
    
    def test_mark_completed(self):
        """Test marking paths as completed."""
        manager = PathManager(max_paths=10)
        
        path1 = GenerationPath(tokens=[1, 2], log_prob=-1.0, path_id=0)
        path2 = GenerationPath(tokens=[3, 4], log_prob=-2.0, path_id=1)
        
        manager.paths = [path1, path2]
        
        manager.mark_completed(path1)
        
        assert len(manager.paths) == 1
        assert len(manager.completed_paths) == 1
        assert manager.completed_paths[0] == path1
    
    def test_get_all_paths(self):
        """Test getting all paths."""
        manager = PathManager(max_paths=10)
        
        path1 = GenerationPath(tokens=[1], log_prob=-1.0, path_id=0)
        path2 = GenerationPath(tokens=[2], log_prob=-2.0, path_id=1)
        path3 = GenerationPath(tokens=[3], log_prob=-3.0, path_id=2)
        
        manager.paths = [path1, path2]
        manager.completed_paths = [path3]
        
        all_paths = manager.get_all_paths()
        
        assert len(all_paths) == 3
        assert path1 in all_paths
        assert path2 in all_paths
        assert path3 in all_paths
    
    def test_clear(self):
        """Test clearing all paths."""
        manager = PathManager(max_paths=10)
        
        manager.paths = [GenerationPath(tokens=[1], log_prob=-1.0, path_id=0)]
        manager.completed_paths = [GenerationPath(tokens=[2], log_prob=-2.0, path_id=1)]
        manager.next_path_id = 5
        
        manager.clear()
        
        assert len(manager.paths) == 0
        assert len(manager.completed_paths) == 0
        assert manager.next_path_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])