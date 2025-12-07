"""
Utility functions for entropy-adaptive branching.

Provides helper functions for seeding, formatting, and analysis.
"""

import torch
import random
import numpy as np
from typing import List, Dict, Any, Optional
import json


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed value
    
    Examples:
        >>> set_seed(42)
        >>> # Now all random operations are reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_results(paths: list, tokenizer, include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    Format generation paths into readable results.
    
    Args:
        paths: List of GenerationPath objects
        tokenizer: Tokenizer to decode tokens
        include_metadata: Whether to include path metadata
    
    Returns:
        List of dictionaries with formatted results
    
    Examples:
        >>> results = format_results(paths, tokenizer)
        >>> for r in results:
        ...     print(r['text'])
        ...     print(f"Probability: {r['probability']:.4f}")
    """
    results = []
    
    for path in paths:
        result = {
            'text': tokenizer.decode(path.tokens, skip_special_tokens=True),
            'tokens': path.tokens,
            'log_prob': path.log_prob,
            'probability': path.probability,
        }
        
        if include_metadata:
            result.update({
                'path_id': path.path_id,
                'parent_id': path.parent_id,
                'length': path.length,
                'branch_points': path.branch_points,
                'num_branches': len(path.branch_points)
            })
        
        results.append(result)
    
    # Sort by probability (descending)
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    return results


def compute_statistics(paths: list) -> Dict[str, Any]:
    """
    Compute statistics over generated paths.
    
    Args:
        paths: List of GenerationPath objects
    
    Returns:
        Dictionary with various statistics
    
    Examples:
        >>> stats = compute_statistics(paths)
        >>> print(f"Average length: {stats['avg_length']:.1f}")
        >>> print(f"Total branches: {stats['total_branches']}")
    """
    if not paths:
        return {}
    
    lengths = [p.length for p in paths]
    log_probs = [p.log_prob for p in paths]
    probs = [p.probability for p in paths]
    num_branches = [len(p.branch_points) for p in paths]
    
    return {
        'num_paths': len(paths),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'avg_log_prob': np.mean(log_probs),
        'std_log_prob': np.std(log_probs),
        'avg_probability': np.mean(probs),
        'total_branches': sum(num_branches),
        'avg_branches_per_path': np.mean(num_branches),
        'max_branches_per_path': np.max(num_branches)
    }


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated texts.
    
    Args:
        texts: List of generated text strings
    
    Returns:
        Dictionary with diversity metrics
    
    Metrics:
        - unique_ratio: Fraction of unique texts
        - avg_edit_distance: Average edit distance between texts
        - vocab_diversity: Number of unique tokens / total tokens
    """
    from collections import Counter
    
    if not texts:
        return {}
    
    # Unique ratio
    unique_texts = len(set(texts))
    unique_ratio = unique_texts / len(texts)
    
    # Vocabulary diversity
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    if all_words:
        vocab_diversity = len(set(all_words)) / len(all_words)
    else:
        vocab_diversity = 0.0
    
    # Average pairwise edit distance (for small sets)
    if len(texts) <= 10:
        try:
            from Levenshtein import distance
            distances = []
            for i, text1 in enumerate(texts):
                for text2 in texts[i+1:]:
                    distances.append(distance(text1, text2))
            avg_edit_distance = np.mean(distances) if distances else 0.0
        except ImportError:
            avg_edit_distance = None
    else:
        avg_edit_distance = None
    
    metrics = {
        'unique_ratio': unique_ratio,
        'num_unique': unique_texts,
        'vocab_diversity': vocab_diversity
    }
    
    if avg_edit_distance is not None:
        metrics['avg_edit_distance'] = avg_edit_distance
    
    return metrics


def save_results(results: List[Dict], filepath: str, format: str = 'json'):
    """
    Save generation results to file.
    
    Args:
        results: List of result dictionaries
        filepath: Path to save file
        format: Output format ('json' or 'txt')
    
    Examples:
        >>> save_results(results, 'outputs/generation_results.json')
    """
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format == 'txt':
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(f"Text: {result['text']}\n")
                f.write(f"Probability: {result['probability']:.6f}\n")
                f.write(f"Log Probability: {result['log_prob']:.4f}\n")
                if 'branch_points' in result:
                    f.write(f"Branch Points: {result['branch_points']}\n")
                f.write("\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")


def load_results(filepath: str, format: str = 'json') -> List[Dict]:
    """
    Load generation results from file.
    
    Args:
        filepath: Path to load file
        format: Input format ('json' or 'txt')
    
    Returns:
        List of result dictionaries
    """
    if format == 'json':
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json'.")


def print_generation_tree(paths: list, tokenizer, max_depth: int = 3):
    """
    Print a visual tree of generation paths.
    
    Args:
        paths: List of GenerationPath objects
        tokenizer: Tokenizer to decode tokens
        max_depth: Maximum depth to display
    
    Examples:
        >>> print_generation_tree(paths, tokenizer)
        Root
        ├── Path 0: "The capital is Paris" (p=0.45)
        ├── Path 1: "The capital of France is Paris" (p=0.30)
        └── Path 2: "Paris is the capital" (p=0.25)
    """
    # Build parent-child relationships
    children = {}
    roots = []
    
    for path in paths:
        if path.parent_id is None:
            roots.append(path)
        else:
            if path.parent_id not in children:
                children[path.parent_id] = []
            children[path.parent_id].append(path)
    
    def print_path(path, prefix="", is_last=True, depth=0):
        if depth > max_depth:
            return
        
        # Decode first few tokens
        preview_tokens = path.tokens[:5]
        preview = tokenizer.decode(preview_tokens, skip_special_tokens=True)
        if len(path.tokens) > 5:
            preview += "..."
        
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}Path {path.path_id}: \"{preview}\" (p={path.probability:.3f})")
        
        # Print children
        if path.path_id in children:
            child_paths = children[path.path_id]
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(child_paths):
                print_path(child, new_prefix, i == len(child_paths) - 1, depth + 1)
    
    print("Generation Tree:")
    for i, root in enumerate(roots):
        print_path(root, "", i == len(roots) - 1, 0)


def visualize_branching_statistics(paths: list, save_path: Optional[str] = None):
    """
    Create visualizations of branching statistics.
    
    Args:
        paths: List of GenerationPath objects
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and seaborn required. Install with: pip install matplotlib seaborn")
        return
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Length distribution
    lengths = [p.length for p in paths]
    axes[0, 0].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Path Length (tokens)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Path Lengths')
    
    # 2. Probability distribution
    probs = [p.probability for p in paths]
    axes[0, 1].hist(probs, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Path Probability')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Path Probabilities')
    axes[0, 1].set_xscale('log')
    
    # 3. Number of branches per path
    num_branches = [len(p.branch_points) for p in paths]
    axes[1, 0].hist(num_branches, bins=range(max(num_branches) + 2), 
                     edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of Branches')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of Branches per Path')
    
    # 4. Length vs Probability scatter
    axes[1, 1].scatter(lengths, probs, alpha=0.6, s=100)
    axes[1, 1].set_xlabel('Path Length (tokens)')
    axes[1, 1].set_ylabel('Path Probability')
    axes[1, 1].set_title('Path Length vs Probability')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def benchmark_generation(eab_generate_fn, naive_generate_fn, 
                        prompts: List[str], num_samples: int = 10) -> Dict:
    """
    Benchmark EAB vs naive generation.
    
    Args:
        eab_generate_fn: Function that generates with EAB
        naive_generate_fn: Function that generates naively
        prompts: List of prompts to test
        num_samples: Number of samples per prompt
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    eab_times = []
    naive_times = []
    
    for prompt in prompts:
        # Benchmark EAB
        start = time.time()
        eab_results = eab_generate_fn(prompt, num_samples)
        eab_times.append(time.time() - start)
        
        # Benchmark naive
        start = time.time()
        naive_results = naive_generate_fn(prompt, num_samples)
        naive_times.append(time.time() - start)
    
    return {
        'eab_avg_time': np.mean(eab_times),
        'naive_avg_time': np.mean(naive_times),
        'speedup': np.mean(naive_times) / np.mean(eab_times),
        'eab_times': eab_times,
        'naive_times': naive_times
    }


class ProgressTracker:
    """Track and display generation progress."""
    
    def __init__(self, total_tokens: int, use_tqdm: bool = True):
        self.total_tokens = total_tokens
        self.current_token = 0
        self.use_tqdm = use_tqdm
        
        if use_tqdm:
            try:
                from tqdm import tqdm
                self.pbar = tqdm(total=total_tokens, desc="Generating")
            except ImportError:
                self.use_tqdm = False
                self.pbar = None
        else:
            self.pbar = None
    
    def update(self, n: int = 1):
        """Update progress by n tokens."""
        self.current_token += n
        if self.pbar:
            self.pbar.update(n)
        elif not self.use_tqdm:
            print(f"Progress: {self.current_token}/{self.total_tokens} tokens", end='\r')
    
    def close(self):
        """Close progress tracker."""
        if self.pbar:
            self.pbar.close()
        elif not self.use_tqdm:
            print()  # New line