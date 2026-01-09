"""
Test script to run quality assessment with example template data.

This script copies the template files and runs the quality assessment
to verify everything works correctly.
"""

import shutil
import json
from pathlib import Path

quality_dir = Path(__file__).parent


def setup_test_data():
    """Copy template files to actual input files."""
    eab_template = quality_dir / "eab_samples_template.json"
    naive_template = quality_dir / "naive_samples_template.json"

    eab_target = quality_dir / "eab_samples.json"
    naive_target = quality_dir / "naive_samples.json"

    print("Setting up test data...")

    # Load templates and expand with more examples
    with open(eab_template, 'r') as f:
        eab_data = json.load(f)

    with open(naive_template, 'r') as f:
        naive_data = json.load(f)

    # Expand each prompt to have more generations (simulate ~20)
    for prompt in eab_data:
        original_gens = eab_data[prompt].copy()
        while len(eab_data[prompt]) < 20:
            # Duplicate with slight variations
            for gen in original_gens[:3]:
                if len(eab_data[prompt]) >= 20:
                    break
                eab_data[prompt].append(gen + " Additionally, this approach ensures long-term success.")

    for prompt in naive_data:
        original_gens = naive_data[prompt].copy()
        while len(naive_data[prompt]) < 20:
            # Duplicate (more repetitive for naive)
            for gen in original_gens[:3]:
                if len(naive_data[prompt]) >= 20:
                    break
                naive_data[prompt].append(gen)

    # Save expanded data
    with open(eab_target, 'w') as f:
        json.dump(eab_data, f, indent=2)

    with open(naive_target, 'w') as f:
        json.dump(naive_data, f, indent=2)

    print(f"✓ Created {eab_target.name} ({len(eab_data)} prompts)")
    print(f"✓ Created {naive_target.name} ({len(naive_data)} prompts)")


def run_assessment():
    """Run the quality assessment."""
    print("\nRunning quality assessment...\n")

    # Import and run main assessment
    from quality_assessment import main
    main()


def cleanup(keep_results=True):
    """Clean up test files."""
    if not keep_results:
        files_to_remove = [
            "eab_samples.json",
            "naive_samples.json",
            "metrics.csv",
            "human_eval_prompts.json",
            "demo_example.txt",
            "self_bleu_comparison.png",
            "distinct_n_comparison.png"
        ]

        print("\nCleaning up test files...")
        for filename in files_to_remove:
            filepath = quality_dir / filename
            if filepath.exists():
                filepath.unlink()
                print(f"  ✓ Removed {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("QUALITY ASSESSMENT TEST RUN")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Create test data from templates")
    print("  2. Run the quality assessment")
    print("  3. Generate all outputs")
    print("\n")

    try:
        setup_test_data()
        run_assessment()

        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nYou can now:")
        print("  • Review the generated files")
        print("  • Replace eab_samples.json and naive_samples.json with your real data")
        print("  • Run: python quality_assessment.py")
        print("\nKeeping test results for your review.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
