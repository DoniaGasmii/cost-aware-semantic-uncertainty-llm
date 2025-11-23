"""
Stage 1: Exact Replication of Paper's Semantic Uncertainty Method
Uses the official semantic_entropy.py implementation with our API
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

# Import official paper's semantic entropy functions
# Option 1: If you've cloned their repo as a submodule
try:
    from semantic_uncertainty_official.uncertainty.uncertainty_measures.semantic_entropy import (
        get_semantic_ids,
        logsumexp_by_id,
        predictive_entropy_rao,
        cluster_assignment_entropy,
        EntailmentDeberta,
        EntailmentGPT4,
        EntailmentGPT35,
    )
except ImportError:
    # Option 2: Manual implementation below (fallback)
    logging.warning("Could not import official semantic_entropy. Using fallback implementation.")
    from fallback_semantic_entropy import (
        get_semantic_ids,
        logsumexp_by_id,
        predictive_entropy_rao,
        cluster_assignment_entropy,
        EntailmentDeberta,
    )


@dataclass
class UncertaintyResult:
    """Results from exact paper replication"""
    # Core metrics (matching paper)
    semantic_entropy: float
    predictive_entropy: float  # Regular entropy
    cluster_assignment_entropy: float
    
    # Metadata
    n_clusters: int
    n_samples: int
    semantic_ids: List[int]
    
    # Raw data
    completions: List[str]
    log_likelihoods: List[List[float]]
    
    # Optional
    weighted_log_likelihoods: List[float] = None


class ExactPaperReplication:
    """
    Stage 1: Exact replication of the paper's method
    
    Uses:
    - Their semantic clustering (get_semantic_ids with entailment)
    - Their entropy calculations (logsumexp_by_id, predictive_entropy_rao)
    - Standard HuggingFace generation (no KV-cache optimization yet)
    
    References:
        Kuhn et al. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation 
        in Natural Language Generation" (2023)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        entailment_model: str = "deberta",  # or "gpt-4", "gpt-3.5"
        device: str = None,
        strict_entailment: bool = True,
        condition_on_question: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model for generation
            entailment_model: Model for semantic clustering
                - "deberta": DeBERTa-v2-xlarge-mnli (local, free)
                - "gpt-4": GPT-4 API (requires OPENAI_API_KEY)
                - "gpt-3.5": GPT-3.5 API
            device: Device for generation model
            strict_entailment: Use bidirectional entailment (paper default: True)
            condition_on_question: Prepend question to answers for clustering
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.strict_entailment = strict_entailment
        self.condition_on_question = condition_on_question
        
        # Load generation model
        logging.info(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load entailment model (for semantic clustering)
        logging.info(f"Loading entailment model: {entailment_model}")
        if entailment_model == "deberta":
            self.entailment_model = EntailmentDeberta()
        elif entailment_model == "gpt-4":
            self.entailment_model = EntailmentGPT4(
                entailment_cache_id=None,
                entailment_cache_only=False
            )
        elif entailment_model == "gpt-3.5":
            self.entailment_model = EntailmentGPT35(
                entailment_cache_id=None,
                entailment_cache_only=False
            )
        else:
            raise ValueError(f"Unknown entailment model: {entailment_model}")
    
    def generate_samples(
        self,
        prompt: str,
        n_samples: int = 10,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generate multiple completions with token log-likelihoods
        
        This matches the paper's generation process:
        - High temperature sampling (default: 1.0)
        - Extract per-token log-likelihoods for entropy calculation
        
        Returns:
            (completions, log_likelihoods_per_completion)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        completions = []
        all_log_likelihoods = []
        
        for i in range(n_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decode completion
            completion = self.tokenizer.decode(
                outputs.sequences[0][prompt_length:],
                skip_special_tokens=True
            )
            completions.append(completion)
            
            # Extract token log-likelihoods (following paper's method)
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True
            )
            log_likelihoods = [score.item() for score in transition_scores[0]]
            all_log_likelihoods.append(log_likelihoods)
        
        return completions, all_log_likelihoods
    
    def estimate_uncertainty(
        self,
        prompt: str,
        question: str = None,  # Optional: for conditioning entailment
        n_samples: int = 10,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> UncertaintyResult:
        """
        Exact replication of paper's semantic uncertainty estimation
        
        Pipeline:
        1. Generate N completions with log-likelihoods
        2. Cluster semantically using entailment model
        3. Calculate three entropy variants:
           - Semantic entropy (main metric in paper)
           - Regular predictive entropy (baseline)
           - Cluster assignment entropy (no likelihoods)
        
        Args:
            prompt: Input prompt for generation
            question: Optional question (for conditioning entailment on question)
            n_samples: Number of completions (paper uses 10-20)
            max_new_tokens: Max tokens per completion
            temperature: Sampling temperature (paper uses 1.0)
        
        Returns:
            UncertaintyResult with all paper metrics
        """
        logging.info(f"Generating {n_samples} completions...")
        
        # Step 1: Generate completions
        completions, log_likelihoods = self.generate_samples(
            prompt=prompt,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Step 2: Prepare completions for clustering
        responses = completions.copy()
        
        # Optionally prepend question (paper does this for DeBERTa)
        if self.condition_on_question and question is not None:
            responses = [f"{question} {r}" for r in responses]
        
        # Step 3: Get semantic clusters via entailment
        logging.info("Clustering semantically with entailment model...")
        
        # Create example dict (needed by paper's API)
        example = {"question": question or prompt}
        
        semantic_ids = get_semantic_ids(
            responses,
            model=self.entailment_model,
            strict_entailment=self.strict_entailment,
            example=example
        )
        
        n_clusters = len(set(semantic_ids))
        logging.info(f"Found {n_clusters} semantic clusters")
        
        # Step 4: Calculate entropy variants (following paper exactly)
        
        # Length-normalized log-likelihoods (average over tokens)
        log_liks_agg = [np.mean(log_lik) for log_lik in log_likelihoods]
        
        # 4a. Regular predictive entropy (baseline)
        regular_entropy = -np.sum(log_liks_agg) / len(log_liks_agg)
        
        # 4b. Semantic entropy (main paper metric)
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids,
            log_liks_agg,
            agg='sum_normalized'
        )
        semantic_entropy = predictive_entropy_rao(log_likelihood_per_semantic_id)
        
        # 4c. Cluster assignment entropy (no likelihoods)
        cluster_entropy = cluster_assignment_entropy(semantic_ids)
        
        result = UncertaintyResult(
            semantic_entropy=semantic_entropy,
            predictive_entropy=regular_entropy,
            cluster_assignment_entropy=cluster_entropy,
            n_clusters=n_clusters,
            n_samples=n_samples,
            semantic_ids=semantic_ids,
            completions=completions,
            log_likelihoods=log_likelihoods,
            weighted_log_likelihoods=log_liks_agg
        )
        
        # Print results (matching paper's logging)
        logging.info(f"\nResults:")
        logging.info(f"  Semantic IDs: {semantic_ids}")
        logging.info(f"  Semantic Entropy: {semantic_entropy:.3f}")
        logging.info(f"  Regular Entropy: {regular_entropy:.3f}")
        logging.info(f"  Cluster Entropy: {cluster_entropy:.3f}")
        logging.info(f"  Num Clusters: {n_clusters}")
        
        return result


def demo():
    """Demo exact paper replication"""
    
    # Initialize with DeBERTa (free, local entailment model)
    estimator = ExactPaperReplication(
        model_name="gpt2",
        entailment_model="deberta",
        strict_entailment=True,  # Paper default
        condition_on_question=True  # Paper default for DeBERTa
    )
    
    # Example 1: Factual question (should have low semantic entropy)
    print("\n" + "="*70)
    print("Example 1: Factual Question (Expected: Low Semantic Entropy)")
    print("="*70)
    
    question = "What is the capital of France?"
    prompt = f"Question: {question}\nAnswer:"
    
    result1 = estimator.estimate_uncertainty(
        prompt=prompt,
        question=question,
        n_samples=10,
        max_new_tokens=5,
        temperature=1.0
    )
    
    print(f"\nSample completions:")
    for i, (comp, sem_id) in enumerate(zip(result1.completions[:5], result1.semantic_ids[:5])):
        print(f"  {i+1}. [Cluster {sem_id}] '{comp.strip()}'")
    
    # Example 2: Ambiguous question (should have higher semantic entropy)
    print("\n" + "="*70)
    print("Example 2: Ambiguous Question (Expected: High Semantic Entropy)")
    print("="*70)
    
    question = "What is the best programming language?"
    prompt = f"Question: {question}\nAnswer:"
    
    result2 = estimator.estimate_uncertainty(
        prompt=prompt,
        question=question,
        n_samples=10,
        max_new_tokens=10,
        temperature=1.0
    )
    
    print(f"\nSample completions:")
    for i, (comp, sem_id) in enumerate(zip(result2.completions[:5], result2.semantic_ids[:5])):
        print(f"  {i+1}. [Cluster {sem_id}] '{comp.strip()}'")
    
    # Compare
    print("\n" + "="*70)
    print("Comparison:")
    print("="*70)
    print(f"Factual Q - Semantic Entropy: {result1.semantic_entropy:.3f}")
    print(f"Ambiguous Q - Semantic Entropy: {result2.semantic_entropy:.3f}")
    print(f"Ratio: {result2.semantic_entropy / result1.semantic_entropy:.2f}x")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()