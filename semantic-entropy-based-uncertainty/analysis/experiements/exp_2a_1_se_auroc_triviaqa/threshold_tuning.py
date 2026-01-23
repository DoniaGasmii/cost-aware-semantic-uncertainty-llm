"""
Threshold Tuning for Semantic Entropy

Runs Experiment 2.A.1 across multiple clustering distance thresholds
to find the optimal setting for TriviaQA.
"""

import sys
import os  # ← ADD THIS IMPORT
import json
import yaml
import subprocess
import shutil
from pathlib import Path
import pandas as pd

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# Add path for config loading
sys.path.insert(0, str(SCRIPT_DIR))

def load_base_config() -> dict:
    """Load base config and return as dict."""
    with open(SCRIPT_DIR / "config.yaml", 'r') as f:
        return yaml.safe_load(f)

def save_modified_config(config: dict, threshold: float):
    """Save a modified config with updated threshold."""
    config = config.copy()
    config['semantic_entropy']['default_threshold'] = threshold
    temp_config_path = SCRIPT_DIR / f"config_temp_{threshold:.2f}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    return temp_config_path

def run_single_threshold(threshold: float):
    """Run full pipeline (experiment → analysis → plotting) for one threshold."""
    print(f"\n{'='*60}")
    print(f" RUNNING THRESHOLD: δ = {threshold:.2f} ")
    print(f"{'='*60}")

    base_config = load_base_config()
    temp_config = save_modified_config(base_config, threshold)

    threshold_str = f"{threshold:.2f}".replace(".", "_")
    output_dir = SCRIPT_DIR / "results" / f"threshold_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config['output']['results_dir'] = str(output_dir.relative_to(SCRIPT_DIR))
    temp_config = save_modified_config(base_config, threshold)

    try:
        # Step 1: Run experiment
        print("→ Running experiment...")
        subprocess.run([
            sys.executable, str(SCRIPT_DIR / "run_experiment.py"),
            "--config", str(temp_config)
        ], cwd=SCRIPT_DIR, env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}, check=True)

        # Step 2: Run analysis
        print("→ Analyzing results...")
        subprocess.run([
            sys.executable, str(SCRIPT_DIR / "analyze_results.py"),
            "--config", str(temp_config)
        ], cwd=SCRIPT_DIR, env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}, check=True)

        # Step 3: Run plotting
        print("→ Generating plots...")
        subprocess.run([
            sys.executable, str(SCRIPT_DIR / "plot_results.py"),
            "--config", str(temp_config)
        ], cwd=SCRIPT_DIR, env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}, check=True)

        print(f"✓ Completed threshold δ = {threshold:.2f}")

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed at threshold δ = {threshold:.2f}: {e}")
    finally:
        if temp_config.exists():
            temp_config.unlink()

    return output_dir

def aggregate_results():
    """Collect AUROC and stats from all thresholds into a summary CSV."""
    results_root = SCRIPT_DIR / "results"
    summary_data = []

    for thresh_dir in sorted(results_root.glob("threshold_*")):
        if not thresh_dir.is_dir():
            continue

        thresh_str = thresh_dir.name.replace("threshold_", "").replace("_", ".")
        threshold = float(thresh_str)

        summary_file = thresh_dir / "summary_stats.json"
        if not summary_file.exists():
            print(f"⚠️  No summary for δ={threshold}")
            continue

        with open(summary_file, 'r') as f:
            stats = json.load(f)

        auroc = None
        if 'auroc_metrics' in stats and 'se_uncertainty_score' in stats['auroc_metrics']:
            auroc = stats['auroc_metrics']['se_uncertainty_score']['best_incorrect'].get('auroc')

        summary_data.append({
            'distance_threshold': threshold,
            'auroc': auroc,
            'accuracy': stats.get('accuracy'),
            'avg_se_uncertainty': stats.get('avg_se'),
            'results_dir': str(thresh_dir.resolve())
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('distance_threshold')
    csv_path = results_root / "threshold_tuning_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Threshold tuning summary saved to: {csv_path}")

    if not df['auroc'].isna().all():
        best_row = df.loc[df['auroc'].idxmax()]
        print(f"\n🏆 Best threshold: δ = {best_row['distance_threshold']:.2f} (AUROC = {best_row['auroc']:.3f})")

    return df

def main():
    print("🚀 STARTING SEMANTIC ENTROPY THRESHOLD TUNING")
    print("   Testing distance thresholds: [0.05, 0.10, 0.15, 0.20, 0.25]")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    for thresh in thresholds:
        run_single_threshold(thresh)

    print("\n" + "="*60)
    print("AGGREGATING RESULTS ACROSS THRESHOLDS")
    print("="*60)
    aggregate_results()

    print("\n✅ Threshold tuning complete!")
    print("   - Raw results per threshold: results/threshold_XX_XX/")
    print("   - Summary CSV for plotting: results/threshold_tuning_summary.csv")


if __name__ == "__main__":
    main()