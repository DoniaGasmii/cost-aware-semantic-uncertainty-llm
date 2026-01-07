"""
Core implementation of Entropy-Adaptive Branching (EAB).

This module provides the main EntropyAdaptiveBranching class that implements
efficient multi-sample generation through entropy-based branching.
"""

import os
import sys

# Handle CUDA library issues gracefully
try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    if "libcudnn" in str(e) or "CUDA" in str(e):
        print("Warning: CUDA libraries not found. Falling back to CPU-only mode.")
        print("If you need GPU support, please install the correct CUDA libraries.")
        # Force CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        import torch.nn.functional as F
    else:
        raise

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import List, Dict, Any, Optional, Tuple
import warnings

from .path import GenerationPath, PathManager
from .entropy import (
    compute_normalized_entropy, 
    should_branch, 
    get_top_k_tokens,
    EntropyTracker
)
from .cache import get_cache_size, CacheManager
from .cache_cow import CopyOnWriteCache, cow_cache_copy
from .utils import set_seed, format_results, compute_statistics, ProgressTracker


class EntropyAdaptiveBranching:
    """
    Main class for entropy-adaptive branching generation.
    
    This implementation efficiently generates multiple diverse samples by:
    1. Encoding the prompt once and caching KV states
    2. Generating tokens autoregressively with shared computation
    3. Branching into multiple paths only when entropy is high
    4. Reusing cached computations across all branches
    
    Examples:
        >>> eab = EntropyAdaptiveBranching("gpt2", entropy_threshold=0.4)
        >>> results = eab.generate("The capital of France is", max_new_tokens=20)
        >>> for r in results:
        ...     print(r['text'], r['probability'])
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-xl",
        device: Optional[str] = None,
        entropy_threshold: float = 0.4,
        branch_factor: int = 3,
        max_paths: int = 20,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize the Entropy-Adaptive Branching system.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            entropy_threshold: Normalized entropy threshold for branching (0-1)
            branch_factor: Number of branches to create when entropy is high
            max_paths: Maximum number of concurrent paths to maintain
            cache_dir: Directory for model cache
            torch_dtype: Data type for model weights (e.g., torch.float16)
            trust_remote_code: Whether to trust remote code (for some models)
        """
        # Device setup
        if device is None:
            # Try to use CUDA if available, but fall back to CPU on any issues
            try:
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception as e:
                print(f"Warning: Error checking CUDA availability: {e}")
                print("Falling back to CPU.")
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        ).to(self.device)
        
        self.model.eval()  # Set to evaluation mode
        
        # Configuration
        self.entropy_threshold = entropy_threshold
        self.branch_factor = branch_factor
        self.max_paths = max_paths
        self.vocab_size = self.model.config.vocab_size
        
        # Managers
        self.cache_manager = CacheManager()
        self.entropy_tracker = EntropyTracker()
        
        print(f"Model loaded successfully!")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Entropy threshold: {self.entropy_threshold}")
        print(f"Branch factor: {self.branch_factor}")
        print(f"Max paths: {self.max_paths}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_prob_threshold: float = 1e-6,
        return_metadata: bool = True,
        show_progress: bool = True,
        seed: Optional[int] = None,
        use_chat_template: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple samples using entropy-adaptive branching.

        Args:
            prompt: Input text prompt (will be formatted as a user message if use_chat_template=True)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling (None = no filtering)
            top_p: Nucleus sampling threshold (None = no filtering)
            min_prob_threshold: Minimum probability to keep a path
            return_metadata: Whether to include path metadata in results
            show_progress: Whether to show progress bar
            seed: Random seed for reproducibility
            use_chat_template: Whether to format prompt using chat template (recommended for instruct models)

        Returns:
            List of dictionaries containing generated text and metadata

        Examples:
            >>> results = eab.generate(
            ...     "What is the capital of France?",
            ...     max_new_tokens=30,
            ...     temperature=0.8,
            ...     use_chat_template=True  # For instruct models
            ... )
        """
        if seed is not None:
            set_seed(seed)
        
        # Reset trackers
        self.entropy_tracker.reset()
        self.cache_manager.reset_statistics()

        # Encode prompt - use chat template for instruct models
        if use_chat_template:
            # Format as a chat message for instruct models
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        else:
            # Raw text encoding
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        prompt_length = input_ids.shape[1]

        print(f"\nPrompt: '{prompt}'")
        if use_chat_template:
            print(f"  (formatted with chat template)")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Generating up to {max_new_tokens} new tokens...")
        
        # Initial forward pass to get prompt cache
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Wrap in Copy-on-Write cache for efficient branching
            if past_key_values is not None:
                if isinstance(past_key_values, tuple):
                    cow_cache = CopyOnWriteCache.from_legacy_cache(past_key_values, device=self.device)
                elif isinstance(past_key_values, DynamicCache):
                    # Convert DynamicCache to legacy, then to COW
                    cow_cache = CopyOnWriteCache.from_legacy_cache(
                        past_key_values.to_legacy_cache(),
                        device=self.device
                    )
                else:
                    cow_cache = CopyOnWriteCache.from_legacy_cache(past_key_values, device=self.device)
            else:
                cow_cache = CopyOnWriteCache(device=self.device)

        # Initialize path manager
        path_manager = PathManager(max_paths=self.max_paths)
        initial_path = path_manager.create_initial_path(cache=cow_cache)
        
        # Progress tracking
        progress = ProgressTracker(max_new_tokens, use_tqdm=show_progress)
        
        # Generation loop
        for position in range(max_new_tokens):
            active_paths = path_manager.get_active_paths()

            if not active_paths:
                break

            # Process each active path
            new_paths = []
            
            for path in active_paths:
                # Get logits for this path
                if position == 0:
                    # First token: use prompt logits
                    path_logits = logits[0]
                    path_cache = cow_cache
                else:
                    # Subsequent tokens: use path's cache
                    last_token = torch.tensor([[path.tokens[-1]]]).to(self.device)

                    # Convert COW cache to format the model expects
                    cache_to_use = path.cache
                    if isinstance(cache_to_use, CopyOnWriteCache):
                        # Get combined cache in legacy format
                        legacy_cache = cache_to_use.to_legacy_cache()
                        if legacy_cache is not None:
                            cache_to_use = DynamicCache.from_legacy_cache(legacy_cache)
                        else:
                            cache_to_use = None
                    elif cache_to_use is not None and isinstance(cache_to_use, tuple):
                        cache_to_use = DynamicCache.from_legacy_cache(cache_to_use)

                    with torch.no_grad():
                        path_outputs = self.model(
                            last_token,
                            past_key_values=cache_to_use,
                            use_cache=True
                        )
                    path_logits = path_outputs.logits[0, -1, :]

                    # Update the COW cache with new KV states
                    new_kv = path_outputs.past_key_values
                    if isinstance(new_kv, DynamicCache):
                        # Extract only the new token's KV (last position)
                        for layer_idx in range(len(new_kv.key_cache)):
                            key = new_kv.key_cache[layer_idx][:, :, -1:, :]
                            value = new_kv.value_cache[layer_idx][:, :, -1:, :]
                            path.cache.update(key, value, layer_idx)
                    path_cache = path.cache
                
                # Apply temperature
                if temperature != 1.0:
                    path_logits = path_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = path_logits < torch.topk(path_logits, top_k)[0][..., -1, None]
                    path_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(path_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    path_logits[indices_to_remove] = float('-inf')
                
                # Compute entropy
                normalized_entropy = compute_normalized_entropy(
                    path_logits,
                    self.vocab_size,
                    temperature=1.0  # Already applied temperature above
                )

                # Decide whether to branch based on entropy only (no hard path limit)
                should_branch_now = normalized_entropy >= self.entropy_threshold

                # Adaptive branch factor: reduce branching as we approach max_paths
                remaining_budget = self.max_paths - len(path_manager)
                if should_branch_now:
                    if remaining_budget >= self.branch_factor:
                        # Plenty of budget: full branching
                        actual_branch_factor = self.branch_factor
                    elif remaining_budget > 0:
                        # Limited budget: partial branching
                        actual_branch_factor = remaining_budget
                    else:
                        # Over budget: minimal branching (2 paths)
                        # Allow exploration of high-entropy positions, rely on pruning
                        actual_branch_factor = 2
                else:
                    actual_branch_factor = 0  # No branching

                # Track entropy
                self.entropy_tracker.record(position, normalized_entropy, should_branch_now)

                if should_branch_now and actual_branch_factor > 0:
                    # Branch into multiple paths with adaptive factor
                    branched_paths = path_manager.branch_path(
                        path,
                        actual_branch_factor,
                        prompt_length + position
                    )

                    # Sample different tokens for each branch
                    probs = F.softmax(path_logits, dim=-1)

                    # Sample multiple tokens at once
                    sampled_tokens = torch.multinomial(probs, actual_branch_factor, replacement=True)

                    for branch_path, token_id in zip(branched_paths, sampled_tokens):
                        token_id = token_id.item()
                        token_log_prob = F.log_softmax(path_logits, dim=-1)[token_id].item()

                        branch_path.add_token(token_id, token_log_prob)
                        branch_path.cache = deep_copy_cache(path_cache)
                        new_paths.append(branch_path)

                else:
                    # Continue with single path (no branching)
                    probs = F.softmax(path_logits, dim=-1)
                    token_id = torch.multinomial(probs, 1).item()
                    token_log_prob = F.log_softmax(path_logits, dim=-1)[token_id].item()

                    path.add_token(token_id, token_log_prob)
                    path.cache = path_cache
                    new_paths.append(path)

            # Check for EOS token in newly generated paths and mark as completed
            paths_to_complete = []
            for new_path in new_paths:
                if new_path.tokens and new_path.tokens[-1] == self.tokenizer.eos_token_id:
                    paths_to_complete.append(new_path)

            for path_to_complete in paths_to_complete:
                path_manager.mark_completed(path_to_complete)
                new_paths.remove(path_to_complete)

            # Update active paths
            path_manager.paths = new_paths

            # Adaptive pruning: if over budget, keep top-k by probability
            if len(path_manager.paths) > self.max_paths:
                # Sort by log probability (descending = highest probability first)
                path_manager.paths.sort(key=lambda p: p.log_prob, reverse=True)
                # Keep only top max_paths
                path_manager.paths = path_manager.paths[:self.max_paths]

            # Additional pruning: remove very low probability paths
            path_manager.prune_paths(min_prob_threshold)
            
            # Update progress
            progress.update(1)
        
        progress.close()

        # Mark any remaining active paths as completed
        active_before = list(path_manager.get_active_paths())  # Make a copy to avoid modification during iteration
        for path in active_before:
            path_manager.mark_completed(path)

        # Get all paths (active + completed)
        all_paths = path_manager.get_all_paths()

        print(f"\nGeneration complete!")
        print(f"Total paths generated: {len(all_paths)}")

        # Compute and print statistics
        stats = compute_statistics(all_paths)
        entropy_stats = self.entropy_tracker.get_statistics()

        print(f"\nStatistics:")
        if stats:  # Only print if we have paths
            print(f"  Average length: {stats['avg_length']:.1f} tokens")
            print(f"  Total branches: {stats['total_branches']}")
        print(f"  Average entropy: {entropy_stats['mean_entropy']:.3f}")
        print(f"  Branch rate: {entropy_stats['branch_rate']:.1%}")
        
        # Format results
        results = format_results(all_paths, self.tokenizer, include_metadata=return_metadata)
        
        return results
    
    def get_entropy_history(self) -> Dict[str, Any]:
        """
        Get entropy tracking history from last generation.
        
        Returns:
            Dictionary with entropy values and branching decisions
        """
        return {
            'positions': self.entropy_tracker.position_history,
            'entropies': self.entropy_tracker.entropy_history,
            'branched': self.entropy_tracker.branching_decisions,
            'statistics': self.entropy_tracker.get_statistics()
        }
    
    def plot_entropy(self, figsize=(12, 6)):
        """
        Plot entropy values from last generation.
        
        Args:
            figsize: Figure size for matplotlib
        """
        self.entropy_tracker.plot(threshold=self.entropy_threshold, figsize=figsize)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get KV-cache usage statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache_manager.get_statistics()
    
    def set_entropy_threshold(self, threshold: float):
        """Update entropy threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Entropy threshold must be in [0, 1]")
        self.entropy_threshold = threshold
        print(f"Entropy threshold updated to: {threshold}")
    
    def set_branch_factor(self, factor: int):
        """Update branch factor."""
        if factor < 2:
            raise ValueError("Branch factor must be >= 2")
        self.branch_factor = factor
        print(f"Branch factor updated to: {factor}")
    
    def set_max_paths(self, max_paths: int):
        """Update maximum number of paths."""
        if max_paths < 1:
            raise ValueError("Max paths must be >= 1")
        self.max_paths = max_paths
        print(f"Max paths updated to: {max_paths}")