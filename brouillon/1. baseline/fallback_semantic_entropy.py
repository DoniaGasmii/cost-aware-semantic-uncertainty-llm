"""
Fallback implementation of semantic_entropy functions
Extracted from the official paper's code for standalone use
Source: https://github.com/lorenzkuhn/semantic_uncertainty
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    """Base class for entailment models"""
    def check_implication(self, text1, text2, *args, **kwargs):
        raise NotImplementedError
    
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """
    DeBERTa-v2-xlarge-mnli for entailment checking
    Returns: 0 (contradiction), 1 (neutral), 2 (entailment)
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(DEVICE)
    
    def check_implication(self, text1, text2, *args, **kwargs):
        """
        Check if text1 entails text2
        
        Args:
            text1: Premise
            text2: Hypothesis
            
        Returns:
            0 (contradiction), 1 (neutral), 2 (entailment)
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        # DeBERTa-mnli returns classes: [contradiction, neutral, entailment]
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        
        return prediction


class EntailmentGPT4(BaseEntailment):
    """GPT-4 based entailment (requires OpenAI API)"""
    def __init__(self, entailment_cache_id=None, entailment_cache_only=False):
        self.name = 'gpt-4'
        self.prediction_cache = {}
        self.entailment_cache_only = entailment_cache_only
    
    def check_implication(self, text1, text2, example=None):
        if example is None:
            example = {"question": ""}
        
        question = example.get('question', '')
        
        prompt = f"""We are evaluating answers to the question "{question}"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? "
        prompt += "Respond with entailment, contradiction, or neutral."
        
        # Requires openai library and API key
        try:
            from openai_utils import predict
            response = predict(prompt, temperature=0.02, model=self.name)
        except ImportError:
            raise ImportError(
                "GPT-4 entailment requires 'openai' library and OPENAI_API_KEY"
            )
        
        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('Unclear response, defaulting to neutral')
            return 1


class EntailmentGPT35(EntailmentGPT4):
    """GPT-3.5 based entailment"""
    def __init__(self, entailment_cache_id=None, entailment_cache_only=False):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'gpt-3.5'


def get_semantic_ids(
    strings_list,
    model,
    strict_entailment=False,
    example=None
):
    """
    Group predictions into semantic clusters using entailment
    
    Algorithm:
    1. For each string, check if it's equivalent to any existing cluster
    2. Two strings are equivalent if they bidirectionally entail each other
    3. If not equivalent to any cluster, create a new cluster
    
    Args:
        strings_list: List of text strings to cluster
        model: Entailment model with check_implication method
        strict_entailment: If True, require bidirectional entailment (2,2)
                          If False, allow neutral+entailment
        example: Optional dict with question context
        
    Returns:
        List of cluster IDs (integers starting from 0)
    """
    
    def are_equivalent(text1, text2):
        """Check if two texts are semantically equivalent"""
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)
        
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        
        if strict_entailment:
            # Both directions must be entailment
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            # No contradiction, and not both neutral
            implications = [implication_1, implication_2]
            semantically_equivalent = (
                0 not in implications and 
                implications != [1, 1]
            )
        
        return semantically_equivalent
    
    # Initialize all cluster IDs as unassigned
    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    
    for i, string1 in enumerate(strings_list):
        # Skip if already assigned
        if semantic_set_ids[i] != -1:
            continue
        
        # Assign new cluster ID
        semantic_set_ids[i] = next_id
        
        # Check remaining strings for equivalence
        for j in range(i + 1, len(strings_list)):
            if are_equivalent(string1, strings_list[j]):
                semantic_set_ids[j] = next_id
        
        next_id += 1
    
    assert -1 not in semantic_set_ids
    
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """
    Aggregate log-likelihoods by semantic cluster
    
    For each cluster, compute logsumexp of all generations in that cluster.
    This gives us P(cluster) = sum of P(generation) for generations in cluster.
    
    Args:
        semantic_ids: List of cluster IDs
        log_likelihoods: List of log-likelihoods (one per generation)
        agg: Aggregation method ('sum_normalized')
        
    Returns:
        List of log-likelihoods per cluster
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    
    log_likelihood_per_semantic_id = []
    
    for uid in unique_ids:
        # Find all generations in this cluster
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        
        if agg == 'sum_normalized':
            # Normalize by total probability mass
            # log P(cluster) = log(sum_i exp(log P(gen_i))) - log(sum_all exp(log P(gen)))
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError(f"Unknown aggregation: {agg}")
        
        log_likelihood_per_semantic_id.append(logsumexp_value)
    
    return log_likelihood_per_semantic_id


def predictive_entropy_rao(log_probs):
    """
    Compute entropy from log probabilities
    
    H = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
    
    This is the "semantic entropy" when log_probs are cluster probabilities.
    
    Args:
        log_probs: Log probabilities (e.g., per cluster)
        
    Returns:
        Entropy value
    """
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """
    Entropy based purely on cluster assignment frequencies
    
    Doesn't use generation likelihoods - just counts how often each
    cluster appears. This is a simpler baseline.
    
    Args:
        semantic_ids: List of cluster IDs
        
    Returns:
        Entropy of cluster distribution
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    
    assert np.isclose(probabilities.sum(), 1)
    
    # Avoid log(0)
    entropy = -np.sum(
        probabilities * np.log(probabilities + 1e-10)
    )
    
    return entropy


def predictive_entropy(log_probs):
    """
    Simple predictive entropy (baseline)
    
    Just averages the log-likelihoods across generations.
    E[-log p(x)] ≈ -1/N sum_i log p(x_i)
    
    Args:
        log_probs: List of log-likelihoods
        
    Returns:
        Average negative log-likelihood
    """
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy


# For compatibility with paper's code
def context_entails_response(context, responses, model):
    """
    Check if context entails the responses (baseline metric)
    
    Args:
        context: Context string
        responses: List of response strings
        model: Entailment model
        
    Returns:
        Average entailment score
    """
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


if __name__ == "__main__":
    # Quick test
    print("Testing semantic clustering...")
    
    model = EntailmentDeberta()
    
    # Test case: semantically similar vs different
    responses = [
        "Paris is the capital of France",
        "The capital of France is Paris",
        "London is the capital of England",
        "Paris is France's capital city"
    ]
    
    semantic_ids = get_semantic_ids(
        responses,
        model=model,
        strict_entailment=True
    )
    
    print(f"\nResponses:")
    for i, (resp, sid) in enumerate(zip(responses, semantic_ids)):
        print(f"  {i+1}. [Cluster {sid}] {resp}")
    
    print(f"\nFound {len(set(semantic_ids))} semantic clusters")
    print(f"Cluster assignment entropy: {cluster_assignment_entropy(semantic_ids):.3f}")