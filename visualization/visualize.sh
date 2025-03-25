#!/bin/bash
# Script to visualize the most recent training log

# Find most recent log file
LATEST_LOG=$(ls -t logs/training_*.log | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training logs found in logs/ directory."
    exit 1
fi

echo "Visualizing metrics from: $LATEST_LOG"

# Create output directory for plots
OUTPUT_DIR="plots/$(basename "$LATEST_LOG" .log)"
mkdir -p "$OUTPUT_DIR"

# Run the visualization script
python visualization/plot_metrics.py --log_file="$LATEST_LOG" --output_dir="$OUTPUT_DIR" --csv="$OUTPUT_DIR/metrics.csv"

echo "Plots generated in: $OUTPUT_DIR"
echo "To view, open: $OUTPUT_DIR/epoch_metrics.png"