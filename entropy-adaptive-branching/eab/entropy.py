"""
Entropy computation and normalization for branching decisions.

Provides functions to compute entropy from probability distributions
and normalize it for use as a branching signal.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Compute entropy of a probability distribution from logits.
    
    Entropy measures the uncertainty in the distribution:
    H(p) = -∑ p(x) log p(x)
    
    High entropy → many tokens have similar probability → model is uncertain
    Low entropy → few tokens dominate → model is confident
    
    Args:
        logits: Raw model logits, shape (vocab_size,) or (batch_size, vocab_size)
        temperature: Temperature for probability distribution (default: 1.0)
                    Higher temperature → more uniform → higher entropy
                    Lower temperature → more peaked → lower entropy
    
    Returns:
        Entropy value (in nats, using natural logarithm)
    
    Examples:
        >>> logits = torch.tensor([2.0, 1.0, 0.5, 0.1])
        >>> entropy = compute_entropy(logits)
        >>> print(f"Entropy: {entropy:.4f}")
    """
    # Ensure logits are 1D
    if logits.dim() > 1:
        logits = logits.squeeze()
    
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)

    # Compute entropy: H = -∑ p(x) log p(x)
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)

    # Handle 0 * -inf case (occurs with heavy filtering like top-k/top-p)
    # When probs[i]=0, we have log_probs[i]=-inf, and 0*-inf=NaN
    # Mathematically, lim_{p→0} p*log(p) = 0, so we replace NaN with 0
    entropy_terms = probs * log_probs
    entropy_terms = torch.where(torch.isnan(entropy_terms), torch.zeros_like(entropy_terms), entropy_terms)
    entropy = -entropy_terms.sum()

    return entropy.item()


def normalize_entropy(entropy: float, vocab_size: int) -> float:
    """
    Normalize entropy to [0, 1] range for consistent thresholding.
    
    Maximum entropy occurs when all tokens are equally likely:
    H_max = log(vocab_size)
    
    Normalized entropy = H / H_max
    
    This allows using the same threshold across different models/vocabularies.
    
    Args:
        entropy: Raw entropy value (in nats)
        vocab_size: Size of the vocabulary
    
    Returns:
        Normalized entropy in [0, 1]
        - 0: Completely certain (one token has probability 1)
        - 1: Maximum uncertainty (uniform distribution)
    
    Examples:
        >>> entropy = 2.5
        >>> vocab_size = 50000
        >>> normalized = normalize_entropy(entropy, vocab_size)
        >>> print(f"Normalized entropy: {normalized:.4f}")
    """
    max_entropy = math.log(vocab_size)
    return min(entropy / max_entropy, 1.0)  # Clamp to [0, 1]


def compute_normalized_entropy(logits: torch.Tensor, vocab_size: int, 
                               temperature: float = 1.0) -> float:
    """
    Compute and normalize entropy in one step.
    
    Convenience function combining compute_entropy and normalize_entropy.
    
    Args:
        logits: Raw model logits
        vocab_size: Size of the vocabulary
        temperature: Temperature for probability distribution
    
    Returns:
        Normalized entropy in [0, 1]
    """
    entropy = compute_entropy(logits, temperature)
    return normalize_entropy(entropy, vocab_size)


def should_branch(logits: torch.Tensor, vocab_size: int, 
                 threshold: float = 0.4, temperature: float = 1.0) -> bool:
    """
    Decide whether to branch based on entropy.
    
    Args:
        logits: Raw model logits
        vocab_size: Size of the vocabulary
        threshold: Entropy threshold for branching (0-1)
        temperature: Temperature for probability distribution
    
    Returns:
        True if entropy exceeds threshold (should branch), False otherwise
    
    Examples:
        >>> logits = torch.randn(50000)
        >>> if should_branch(logits, vocab_size=50000, threshold=0.4):
        ...     print("High uncertainty detected - branching!")
    """
    normalized_entropy = compute_normalized_entropy(logits, vocab_size, temperature)
    return normalized_entropy >= threshold


def get_top_k_tokens(logits: torch.Tensor, k: int = 5, 
                     temperature: float = 1.0) -> tuple:
    """
    Get top-k most probable tokens and their probabilities.
    
    Useful for understanding what the model is considering when branching.
    
    Args:
        logits: Raw model logits
        k: Number of top tokens to return
        temperature: Temperature for probability distribution
    
    Returns:
        Tuple of (token_ids, probabilities, log_probabilities)
    
    Examples:
        >>> logits = torch.randn(50000)
        >>> token_ids, probs, log_probs = get_top_k_tokens(logits, k=3)
        >>> for tid, p in zip(token_ids, probs):
        ...     print(f"Token {tid}: {p:.4f}")
    """
    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature
    
    # Get top-k logits and indices
    top_logits, top_indices = torch.topk(logits, k=k, dim=-1)
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    top_probs = probs[top_indices]
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    top_log_probs = log_probs[top_indices]
    
    return top_indices, top_probs, top_log_probs


def compute_predictive_entropy(samples: list, tokenizer) -> float:
    """
    Compute predictive entropy across multiple generated samples.
    
    This is useful for uncertainty quantification after generation.
    Measures how diverse the generated samples are.
    
    Args:
        samples: List of generated text samples
        tokenizer: Tokenizer to use for encoding
    
    Returns:
        Predictive entropy value
    
    Note:
        This is different from per-token entropy. It measures uncertainty
        over the entire distribution of generated sequences.
    """
    if not samples:
        return 0.0
    
    # Count unique samples
    from collections import Counter
    sample_counts = Counter(samples)
    
    # Compute probabilities
    total = len(samples)
    probs = torch.tensor([count / total for count in sample_counts.values()])
    
    # Compute entropy
    log_probs = torch.log(probs)
    entropy = -(probs * log_probs).sum()
    
    return entropy.item()


class EntropyTracker:
    """
    Track entropy values during generation for analysis and visualization.
    """
    
    def __init__(self):
        self.entropy_history = []
        self.position_history = []
        self.branching_decisions = []
    
    def record(self, position: int, entropy: float, branched: bool):
        """
        Record entropy value and branching decision at a position.
        
        Args:
            position: Token position
            entropy: Entropy value
            branched: Whether branching occurred at this position
        """
        self.position_history.append(position)
        self.entropy_history.append(entropy)
        self.branching_decisions.append(branched)
    
    def get_statistics(self) -> dict:
        """
        Compute statistics over recorded entropy values.
        
        Returns:
            Dictionary with mean, std, min, max entropy values
        """
        if not self.entropy_history:
            return {}
        
        import numpy as np
        entropies = np.array(self.entropy_history)
        
        return {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'min_entropy': float(np.min(entropies)),
            'max_entropy': float(np.max(entropies)),
            'num_branches': sum(self.branching_decisions),
            'branch_rate': sum(self.branching_decisions) / len(self.branching_decisions)
        }
    
    def reset(self):
        """Clear all recorded data."""
        self.entropy_history = []
        self.position_history = []
        self.branching_decisions = []
    
    def plot(self, threshold: Optional[float] = None, figsize=(12, 6)):
        """
        Plot entropy values over generation positions.
        
        Args:
            threshold: Optional threshold line to display
            figsize: Figure size for matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot entropy values
        ax.plot(self.position_history, self.entropy_history, 
                marker='o', linewidth=2, markersize=6, label='Entropy')
        
        # Mark branching points
        branch_positions = [pos for pos, branched in 
                          zip(self.position_history, self.branching_decisions) 
                          if branched]
        branch_entropies = [ent for ent, branched in 
                          zip(self.entropy_history, self.branching_decisions) 
                          if branched]
        
        if branch_positions:
            ax.scatter(branch_positions, branch_entropies, 
                      color='red', s=200, marker='*', 
                      label='Branch Points', zorder=5)
        
        # Add threshold line if provided
        if threshold is not None:
            ax.axhline(y=threshold, color='gray', linestyle='--', 
                      linewidth=2, label=f'Threshold ({threshold})')
        
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Normalized Entropy', fontsize=12)
        ax.set_title('Entropy Values During Generation', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()