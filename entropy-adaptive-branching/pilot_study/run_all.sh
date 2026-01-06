#!/bin/bash
# Complete Pilot Study Pipeline
# Runs experiment, generates visualizations, and provides threshold recommendation

set -e  # Exit on error

echo "================================================================================"
echo "EAB Pilot Study - Complete Pipeline"
echo "================================================================================"
echo ""
echo "This will run the complete pilot study pipeline:"
echo "  1. Run 200 prompts (~20-30 minutes)"
echo "  2. Generate visualizations"
echo "  3. Provide threshold recommendation"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."
echo ""

# Step 1: Run pilot experiment
echo "================================================================================"
echo "STEP 1/3: Running Pilot Experiment"
echo "================================================================================"
python3 run_pilot.py

if [ $? -ne 0 ]; then
    echo "Error: Pilot experiment failed!"
    exit 1
fi

# Step 2: Generate visualizations
echo ""
echo "================================================================================"
echo "STEP 2/3: Generating Visualizations"
echo "================================================================================"
python3 threshold/visualize_distributions.py

if [ $? -ne 0 ]; then
    echo "Error: Visualization generation failed!"
    exit 1
fi

# Step 3: Threshold recommendation
echo ""
echo "================================================================================"
echo "STEP 3/3: Threshold Recommendation"
echo "================================================================================"
python3 threshold/recommend_threshold.py

if [ $? -ne 0 ]; then
    echo "Error: Threshold recommendation failed!"
    exit 1
fi

# Summary
echo ""
echo "================================================================================"
echo "PILOT STUDY COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  📊 Data: results/"
echo "  📈 Plots: plots/"
echo "  📄 Recommendation: results/threshold_recommendation.txt"
echo ""
echo "Next steps:"
echo "  1. Review plots in plots/"
echo "  2. Read threshold recommendation in results/threshold_recommendation.txt"
echo "  3. Select threshold for your experiments"
echo "  4. Proceed to exp_1a with chosen threshold"
echo ""
echo "================================================================================"
