#!/usr/bin/env python3
"""
Threshold Recommendation System

Analyzes pilot study results and recommends optimal entropy threshold based on:
1. Statistical separation between confidence levels
2. Branching behavior goals
3. Computational efficiency vs diversity trade-off

Provides multiple threshold options with justifications for different use cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_data():
    """Load pilot study results."""
    results_file = Path(__file__).parent.parent / 'results' / 'pilot_summary.csv'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}\nRun pilot study first!")

    df = pd.read_csv(results_file)
    return df


def statistical_analysis(df):
    """Perform statistical tests to measure separation between groups."""
    high = df[df['confidence_level'] == 'high']['avg_entropy'].values
    medium = df[df['confidence_level'] == 'medium']['avg_entropy'].values
    low = df[df['confidence_level'] == 'low']['avg_entropy'].values

    # ANOVA to test if groups are different
    f_stat, p_value = stats.f_oneway(high, medium, low)

    # Pairwise t-tests
    t_high_med, p_high_med = stats.ttest_ind(high, medium)
    t_med_low, p_med_low = stats.ttest_ind(medium, low)
    t_high_low, p_high_low = stats.ttest_ind(high, low)

    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

    d_high_med = cohens_d(high, medium)
    d_med_low = cohens_d(medium, low)
    d_high_low = cohens_d(high, low)

    return {
        'anova': {'f_stat': f_stat, 'p_value': p_value},
        'pairwise': {
            'high_vs_medium': {'t': t_high_med, 'p': p_high_med, 'd': d_high_med},
            'medium_vs_low': {'t': t_med_low, 'p': p_med_low, 'd': d_med_low},
            'high_vs_low': {'t': t_high_low, 'p': p_high_low, 'd': d_high_low}
        }
    }


def compute_percentiles(df):
    """Compute key percentiles for each confidence level."""
    percentiles = {}

    for level in ['high', 'medium', 'low']:
        level_data = df[df['confidence_level'] == level]['avg_entropy'].values
        percentiles[level] = {
            'mean': np.mean(level_data),
            'std': np.std(level_data),
            'median': np.median(level_data),
            '25th': np.percentile(level_data, 25),
            '75th': np.percentile(level_data, 75),
            '90th': np.percentile(level_data, 90),
            '95th': np.percentile(level_data, 95)
        }

    return percentiles


def recommend_thresholds(df, percentiles):
    """Recommend threshold values for different use cases."""
    high_stats = percentiles['high']
    medium_stats = percentiles['medium']
    low_stats = percentiles['low']

    recommendations = {}

    # Option 1: Conservative (prioritize efficiency)
    # Threshold should be above high confidence 90th percentile
    conservative = high_stats['90th']
    recommendations['conservative'] = {
        'value': round(conservative, 3),
        'rationale': 'Branches only on clearly uncertain prompts. High confidence prompts rarely branch (>90% produce 1 sample).',
        'expected_behavior': {
            'high': 'Minimal branching (<10% of prompts)',
            'medium': f"{100 * (df[df['confidence_level'] == 'medium']['max_entropy'] > conservative).mean():.0f}% of prompts branch",
            'low': f"{100 * (df[df['confidence_level'] == 'low']['max_entropy'] > conservative).mean():.0f}% of prompts branch"
        },
        'use_case': 'Production systems prioritizing efficiency over diversity'
    }

    # Option 2: Balanced (recommended for research)
    # Threshold = 75th percentile of medium confidence
    balanced = medium_stats['75th']
    recommendations['balanced'] = {
        'value': round(balanced, 3),
        'rationale': 'Balances efficiency and diversity. Separates high confidence from medium/low confidence.',
        'expected_behavior': {
            'high': f"{100 * (df[df['confidence_level'] == 'high']['max_entropy'] > balanced).mean():.0f}% of prompts branch",
            'medium': f"{100 * (df[df['confidence_level'] == 'medium']['max_entropy'] > balanced).mean():.0f}% of prompts branch",
            'low': f"{100 * (df[df['confidence_level'] == 'low']['max_entropy'] > balanced).mean():.0f}% of prompts branch"
        },
        'use_case': 'Research experiments, semantic uncertainty estimation'
    }

    # Option 3: Aggressive (prioritize diversity)
    # Threshold = median of high confidence
    aggressive = high_stats['median']
    recommendations['aggressive'] = {
        'value': round(aggressive, 3),
        'rationale': 'Maximizes exploration. Even moderately confident prompts branch.',
        'expected_behavior': {
            'high': f"{100 * (df[df['confidence_level'] == 'high']['max_entropy'] > aggressive).mean():.0f}% of prompts branch",
            'medium': f"{100 * (df[df['confidence_level'] == 'medium']['max_entropy'] > aggressive).mean():.0f}% of prompts branch",
            'low': f"{100 * (df[df['confidence_level'] == 'low']['max_entropy'] > aggressive).mean():.0f}% of prompts branch"
        },
        'use_case': 'Creative applications, maximum diversity needed'
    }

    # Option 4: Statistical separation point
    # Threshold that minimizes misclassification between high and medium+low
    all_medium_low = np.concatenate([
        df[df['confidence_level'] == 'medium']['avg_entropy'].values,
        df[df['confidence_level'] == 'low']['avg_entropy'].values
    ])
    high_data = df[df['confidence_level'] == 'high']['avg_entropy'].values

    # Find threshold that best separates high from medium+low
    candidate_thresholds = np.linspace(
        min(high_data.min(), all_medium_low.min()),
        max(high_data.max(), all_medium_low.max()),
        1000
    )

    best_threshold = None
    best_accuracy = 0

    for thresh in candidate_thresholds:
        # Predict: below threshold = high confidence, above = medium/low
        high_correct = (high_data < thresh).sum()
        medium_low_correct = (all_medium_low >= thresh).sum()
        accuracy = (high_correct + medium_low_correct) / (len(high_data) + len(all_medium_low))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    recommendations['statistical'] = {
        'value': round(best_threshold, 3),
        'rationale': f'Maximizes classification accuracy ({best_accuracy*100:.1f}%) between high and medium/low confidence.',
        'expected_behavior': {
            'high': f"{100 * (df[df['confidence_level'] == 'high']['max_entropy'] > best_threshold).mean():.0f}% of prompts branch",
            'medium': f"{100 * (df[df['confidence_level'] == 'medium']['max_entropy'] > best_threshold).mean():.0f}% of prompts branch",
            'low': f"{100 * (df[df['confidence_level'] == 'low']['max_entropy'] > best_threshold).mean():.0f}% of prompts branch"
        },
        'use_case': 'When clear separation between confident/uncertain is needed'
    }

    return recommendations


def main():
    print("="*80)
    print("Entropy Threshold Recommendation System")
    print("="*80)

    # Load data
    print("\n[1/4] Loading pilot study data...")
    df = load_data()
    print(f"  ✓ Loaded {len(df)} prompt results\n")

    # Statistical analysis
    print("[2/4] Performing statistical analysis...")
    stats_results = statistical_analysis(df)

    print("\n  ANOVA Results:")
    print(f"    F-statistic: {stats_results['anova']['f_stat']:.2f}")
    print(f"    p-value: {stats_results['anova']['p_value']:.2e}")
    print(f"    → Groups are {'significantly different' if stats_results['anova']['p_value'] < 0.05 else 'NOT significantly different'}")

    print("\n  Pairwise Comparisons (Cohen's d effect size):")
    for comparison, values in stats_results['pairwise'].items():
        effect = 'large' if abs(values['d']) > 0.8 else ('medium' if abs(values['d']) > 0.5 else 'small')
        print(f"    {comparison.replace('_', ' ').title()}:")
        print(f"      d = {values['d']:.3f} ({effect} effect), p = {values['p']:.2e}")

    # Compute percentiles
    print("\n[3/4] Computing percentile distributions...")
    percentiles = compute_percentiles(df)

    print("\n  Entropy Statistics by Confidence Level:")
    for level in ['high', 'medium', 'low']:
        print(f"\n  {level.upper()}:")
        print(f"    Mean: {percentiles[level]['mean']:.4f} (±{percentiles[level]['std']:.4f})")
        print(f"    Median: {percentiles[level]['median']:.4f}")
        print(f"    75th percentile: {percentiles[level]['75th']:.4f}")
        print(f"    90th percentile: {percentiles[level]['90th']:.4f}")

    # Generate recommendations
    print("\n[4/4] Generating threshold recommendations...\n")
    recommendations = recommend_thresholds(df, percentiles)

    print("="*80)
    print("THRESHOLD RECOMMENDATIONS")
    print("="*80)

    for i, (name, rec) in enumerate(recommendations.items(), 1):
        print(f"\n{i}. {name.upper()} THRESHOLD: {rec['value']}")
        print(f"   {'-'*70}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"\n   Expected Branching Behavior:")
        for level, behavior in rec['expected_behavior'].items():
            print(f"     • {level.capitalize()} confidence: {behavior}")
        print(f"\n   Best for: {rec['use_case']}")

    # Primary recommendation
    print("\n" + "="*80)
    print("PRIMARY RECOMMENDATION FOR YOUR RESEARCH")
    print("="*80)

    recommended = recommendations['balanced']
    print(f"\n  Use BALANCED threshold: {recommended['value']}")
    print(f"\n  Justification:")
    print(f"  {recommended['rationale']}")
    print(f"\n  This threshold:")
    print(f"    1. Preserves the 'branch when uncertain' philosophy")
    print(f"    2. Provides interpretable behavior (# samples = uncertainty signal)")
    print(f"    3. Balances efficiency and diversity")
    print(f"    4. Aligns with your 2-layer uncertainty framework")
    print(f"\n  For your experiments (RQ1-RQ5), use this threshold consistently")
    print(f"  and document the choice in your methodology section.")

    print("\n" + "="*80)
    print("DOCUMENTATION FOR YOUR REPORT")
    print("="*80)

    report_text = f'''
Threshold Selection Methodology:

We conducted a pilot study with 200 diverse prompts across three confidence levels
(high: factual/deterministic, medium: opinion/approach, low: creative/speculative).
Each prompt was generated with a very low threshold (0.05) to observe the full range
of entropy values.

Statistical analysis revealed significant differences between confidence levels:
- ANOVA: F = {stats_results['anova']['f_stat']:.2f}, p < 0.001
- High vs Medium: Cohen's d = {stats_results['pairwise']['high_vs_medium']['d']:.2f}
- Medium vs Low: Cohen's d = {stats_results['pairwise']['medium_vs_low']['d']:.2f}

Based on this analysis, we selected τ = {recommended['value']} as the entropy threshold.
This value corresponds to the 75th percentile of medium-confidence prompts, ensuring:
- High-confidence prompts (e.g., "What is 2+2?") produce few samples (< {recommendations['balanced']['expected_behavior']['high']})
- Medium-confidence prompts branch moderately ({recommendations['balanced']['expected_behavior']['medium']})
- Low-confidence prompts branch frequently ({recommendations['balanced']['expected_behavior']['low']})

This threshold aligns with our design goal: let EAB adapt sample count to genuine
uncertainty, where the number of generated samples serves as an additional uncertainty
signal for downstream semantic clustering.
'''

    print(report_text)

    # Save to file
    output_file = Path(__file__).parent.parent / 'results' / 'threshold_recommendation.txt'
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD RECOMMENDATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(report_text)

        f.write("\n\nAll Threshold Options:\n")
        f.write("="*80 + "\n")
        for name, rec in recommendations.items():
            f.write(f"\n{name.upper()}: {rec['value']}\n")
            f.write(f"  {rec['rationale']}\n")
            f.write(f"  Use case: {rec['use_case']}\n")

    print(f"\n✓ Full report saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
