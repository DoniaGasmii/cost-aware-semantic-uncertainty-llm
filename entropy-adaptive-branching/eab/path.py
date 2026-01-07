"""
Generation path tracking for entropy-adaptive branching.

Each GenerationPath represents an independent generation trajectory,
maintaining its own token sequence, probability, and KV-cache state.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class GenerationPath:
    """
    Represents a single generation path in the branching tree.
    
    Attributes:
        tokens: List of generated token IDs (excluding prompt)
        log_prob: Cumulative log probability of this path
        cache: KV-cache state (past_key_values from transformers)
        branch_points: List of positions where this path was created via branching
        parent_id: ID of the parent path (None for initial path)
        path_id: Unique identifier for this path
    """
    tokens: List[int]
    log_prob: float
    cache: Optional[Tuple] = None
    branch_points: List[int] = None
    parent_id: Optional[int] = None
    path_id: Optional[int] = None
    
    def __post_init__(self):
        if self.branch_points is None:
            self.branch_points = []
    
    def copy(self, new_path_id: Optional[int] = None) -> 'GenerationPath':
        """
        Create a deep copy of this path for branching.
        
        Args:
            new_path_id: ID to assign to the new path

        Returns:
            New GenerationPath instance with copied state
        """
        from .cache import deep_copy_cache
        from .cache_cow import CopyOnWriteCache

        # Handle different cache types
        copied_cache = None
        if self.cache is not None:
            if isinstance(self.cache, CopyOnWriteCache):
                # Use COW branching for COW caches
                copied_cache = self.cache.branch()
            else:
                # Use deep copy for regular caches
                copied_cache = deep_copy_cache(self.cache)

        return GenerationPath(
            tokens=self.tokens.copy(),
            log_prob=self.log_prob,
            cache=copied_cache,
            branch_points=self.branch_points.copy(),
            parent_id=self.path_id,
            path_id=new_path_id
        )
    
    def add_token(self, token_id: int, token_log_prob: float):
        """
        Add a new token to this path.
        
        Args:
            token_id: Token ID to add
            token_log_prob: Log probability of this token
        """
        self.tokens.append(token_id)
        self.log_prob += token_log_prob
    
    def mark_branch(self, position: int):
        """
        Mark that this path was created via branching at a specific position.
        
        Args:
            position: Token position where branching occurred
        """
        if position not in self.branch_points:
            self.branch_points.append(position)
    
    @property
    def probability(self) -> float:
        """Convert log probability to actual probability."""
        return torch.exp(torch.tensor(self.log_prob)).item()
    
    @property
    def length(self) -> int:
        """Number of generated tokens in this path."""
        return len(self.tokens)
    
    def __repr__(self) -> str:
        return (f"GenerationPath(path_id={self.path_id}, "
                f"length={self.length}, "
                f"log_prob={self.log_prob:.4f}, "
                f"prob={self.probability:.4f}, "
                f"branches={len(self.branch_points)})")


class PathManager:
    """
    Manages multiple generation paths during adaptive branching.
    
    Handles path creation, pruning, and tracking of the generation tree.
    """
    
    def __init__(self, max_paths: int = 20):
        """
        Initialize the path manager.
        
        Args:
            max_paths: Maximum number of active paths to maintain
        """
        self.max_paths = max_paths
        self.paths: List[GenerationPath] = []
        self.next_path_id = 0
        self.completed_paths: List[GenerationPath] = []
    
    def create_initial_path(self, cache: Optional[Tuple] = None) -> GenerationPath:
        """
        Create the initial generation path.
        
        Args:
            cache: Initial KV-cache from prompt encoding
            
        Returns:
            New GenerationPath instance
        """
        path = GenerationPath(
            tokens=[],
            log_prob=0.0,
            cache=cache,
            path_id=self.next_path_id
        )
        self.next_path_id += 1
        self.paths.append(path)
        return path
    
    def branch_path(self, path: GenerationPath, branch_factor: int, 
                    position: int) -> List[GenerationPath]:
        """
        Create multiple branches from a single path.
        
        Args:
            path: Path to branch from
            branch_factor: Number of branches to create
            position: Token position where branching occurs
            
        Returns:
            List of new GenerationPath instances
        """
        new_paths = []
        for _ in range(branch_factor):
            new_path = path.copy(new_path_id=self.next_path_id)
            new_path.mark_branch(position)
            self.next_path_id += 1
            new_paths.append(new_path)
        
        return new_paths
    
    def prune_paths(self, min_prob_threshold: float = 1e-6):
        """
        Remove low-probability paths to manage memory.

        Args:
            min_prob_threshold: Minimum probability to keep a path
        """
        if len(self.paths) <= self.max_paths:
            return

        # Sort by probability (descending)
        self.paths.sort(key=lambda p: p.log_prob, reverse=True)

        # Keep top max_paths
        self.paths = self.paths[:self.max_paths]

        # Optionally filter out paths below minimum threshold
        # But only if we have enough paths remaining
        filtered_paths = [p for p in self.paths if p.probability >= min_prob_threshold]
        if len(filtered_paths) > 0:
            self.paths = filtered_paths
    
    def mark_completed(self, path: GenerationPath):
        """
        Move a path to completed list.
        
        Args:
            path: Path that has finished generation
        """
        if path in self.paths:
            self.paths.remove(path)
        self.completed_paths.append(path)
    
    def get_active_paths(self) -> List[GenerationPath]:
        """Get all currently active paths."""
        return self.paths
    
    def get_completed_paths(self) -> List[GenerationPath]:
        """Get all completed paths."""
        return self.completed_paths
    
    def get_all_paths(self) -> List[GenerationPath]:
        """Get all paths (active and completed)."""
        return self.paths + self.completed_paths
    
    def clear(self):
        """Clear all paths and reset state."""
        self.paths = []
        self.completed_paths = []
        self.next_path_id = 0
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __repr__(self) -> str:
        return (f"PathManager(active={len(self.paths)}, "
                f"completed={len(self.completed_paths)}, "
                f"max_paths={self.max_paths})")