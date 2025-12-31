"""
Data utilities for saving and loading experimental results.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2):
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation level
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]):
    """
    Save data to pickle file.

    Args:
        data: Object to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(
    results: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    filename: str = "raw_results.json"
):
    """
    Save experimental results to JSON file.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        filename: Name of output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename

    # Add metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(results)
        },
        'results': results
    }

    save_json(output, filepath)
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load experimental results from JSON file.

    Args:
        filepath: Path to results file

    Returns:
        List of result dictionaries
    """
    data = load_json(filepath)
    return data.get('results', [])


def append_result(
    result: Dict[str, Any],
    filepath: Union[str, Path]
):
    """
    Append a single result to existing results file.

    Args:
        result: Result dictionary to append
        filepath: Path to results file
    """
    filepath = Path(filepath)

    # Load existing results or create new
    if filepath.exists():
        data = load_json(filepath)
        results = data.get('results', [])
    else:
        results = []

    # Append new result
    results.append(result)

    # Update metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(results),
            'last_updated': datetime.now().isoformat()
        },
        'results': results
    }

    save_json(output, filepath)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Summary statistics
    """
    import numpy as np

    # Group results by prompt length
    by_length = {}
    for result in results:
        length = result['prompt_length']
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(result)

    # Compute statistics for each length
    summary = {}
    for length, length_results in by_length.items():
        # Extract metrics
        speedup_token = [r['efficiency']['speedup_token_steps'] for r in length_results]
        speedup_time = [r['efficiency']['speedup_time'] for r in length_results]

        summary[length] = {
            'num_prompts': len(length_results),
            'speedup_token_steps': {
                'mean': np.mean(speedup_token),
                'std': np.std(speedup_token),
                'median': np.median(speedup_token),
                'min': np.min(speedup_token),
                'max': np.max(speedup_token)
            },
            'speedup_time': {
                'mean': np.mean(speedup_time),
                'std': np.std(speedup_time),
                'median': np.median(speedup_time),
                'min': np.min(speedup_time),
                'max': np.max(speedup_time)
            }
        }

    return summary
