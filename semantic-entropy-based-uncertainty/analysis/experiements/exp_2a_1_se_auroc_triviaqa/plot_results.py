"""
Plot results from Experiment 2.A.1 with pastel colors and clear labels.
"""

import sys
import json
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import roc_curve, auc

# Set pastel style
sns.set_palette("pastel")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

experiment_dir = Path(__file__).parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = experiment_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(config: Dict[str, Any]):
    results_dir = experiment_dir / config['output']['results_dir']
    with open(results_dir / "raw_results.json") as f:
        return json.load(f)['results']


def plot_roc_curve(results, save_dir):
    se = np.array([r['se_uncertainty_score'] for r in results])
    y_true = 1 - np.array([r['best_sample_correct'] for r in results])
    fpr, tpr, _ = roc_curve(y_true, se)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color=sns.color_palette()[0], linewidth=2.5, label=f'SE (AUROC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Semantic Entropy → Incorrect Answers')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_distribution_comparison(results, save_dir):
    correct = [r['se_uncertainty_score'] for r in results if r['best_sample_correct']]
    incorrect = [r['se_uncertainty_score'] for r in results if not r['best_sample_correct']]
    plt.figure(figsize=(8, 5))
    plt.hist(correct, bins=20, alpha=0.7, label=f'Correct (n={len(correct)})',
             color=sns.color_palette("pastel")[2])
    plt.hist(incorrect, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect)})',
             color=sns.color_palette("pastel")[3])
    plt.axvline(np.mean(correct), color='darkgreen', linestyle='--', linewidth=2)
    plt.axvline(np.mean(incorrect), color='darkred', linestyle='--', linewidth=2)
    plt.xlabel('Semantic Uncertainty Score')
    plt.ylabel('Frequency')
    plt.title('SE Distribution by Answer Correctness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Experiment 2.A.1 results")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    results = load_results(config)
    results_dir = experiment_dir / config['output']['results_dir']
    save_dir = results_dir / "figures"
    save_dir.mkdir(exist_ok=True, parents=True)
    print("Generating plots...")
    plot_roc_curve(results, save_dir)
    plot_distribution_comparison(results, save_dir)
    print(f"All figures saved to: {save_dir}")


if __name__ == "__main__":
    main()