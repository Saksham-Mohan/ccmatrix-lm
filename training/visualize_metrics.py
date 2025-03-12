#!/usr/bin/env python3
"""
Script to visualize training metrics and loss curves from TensorBoard logs.
"""

import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob

def extract_metrics_from_logs(log_dir):
    """Extract metrics from TensorBoard log files."""
    # Find all event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        raise ValueError(f"No event files found in {log_dir}")
    
    # Extract metrics from each file
    all_metrics = {}
    
    for event_file in event_files:
        try:
            for e in tf.compat.v1.train.summary_iterator(event_file):
                for v in e.summary.value:
                    # Get tag name and value
                    tag = v.tag
                    step = e.step
                    
                    # Different TensorFlow versions store values differently
                    if hasattr(v, 'simple_value'):
                        value = v.simple_value
                    elif hasattr(v, 'tensor'):
                        value = tf.make_ndarray(v.tensor).item()
                    else:
                        continue
                    
                    # Store in our metrics dictionary
                    if tag not in all_metrics:
                        all_metrics[tag] = []
                    
                    all_metrics[tag].append((step, value))
        except Exception as e:
            print(f"Error processing file {event_file}: {e}")
    
    # Sort metrics by step
    for tag in all_metrics:
        all_metrics[tag].sort(key=lambda x: x[0])
    
    return all_metrics

def plot_metrics(metrics, output_dir, smooth_factor=0.8):
    """Plot metrics and save figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each metric
    for metric_name, values in metrics.items():
        if not values:
            continue
        
        # Convert to numpy arrays
        steps, metric_values = zip(*values)
        steps = np.array(steps)
        metric_values = np.array(metric_values)
        
        # Apply exponential moving average for smoothing
        smoothed_values = metric_values.copy()
        if len(metric_values) > 1 and smooth_factor > 0:
            for i in range(1, len(metric_values)):
                smoothed_values[i] = smooth_factor * smoothed_values[i-1] + (1-smooth_factor) * metric_values[i]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(steps, metric_values, 'b-', alpha=0.3, label=f'{metric_name} (raw)')
        plt.plot(steps, smoothed_values, 'r-', linewidth=2, label=f'{metric_name} (smoothed)')
        
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over Training')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in metric_name)
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined plots for related metrics
    # Loss plots
    loss_metrics = {k: v for k, v in metrics.items() if 'loss' in k.lower()}
    if loss_metrics:
        plt.figure(figsize=(12, 6))
        for name, values in loss_metrics.items():
            steps, values = zip(*values)
            plt.plot(steps, values, label=name)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Loss Metrics')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "combined_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Perplexity plots
    perplexity_metrics = {k: v for k, v in metrics.items() if 'perplexity' in k.lower()}
    if perplexity_metrics:
        plt.figure(figsize=(12, 6))
        for name, values in perplexity_metrics.items():
            steps, values = zip(*values)
            plt.plot(steps, values, label=name)
        plt.xlabel('Step')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Metrics')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "combined_perplexity.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")
    return loss_metrics, perplexity_metrics

def create_metrics_table(metrics, output_file):
    """Create a CSV table of metrics."""
    # Prepare data for DataFrame
    data = []
    for metric_name, values in metrics.items():
        for step, value in values:
            data.append({
                'metric': metric_name,
                'step': step,
                'value': value
            })
    
    # Create DataFrame and pivot to wide format
    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index='step', columns='metric', values='value')
    
    # Save to CSV
    pivot_df.to_csv(output_file)
    print(f"Metrics table saved to {output_file}")
    return pivot_df

def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics from TensorBoard logs")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing TensorBoard logs")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--smooth", type=float, default=0.8, help="Smoothing factor (0-1) for metrics")
    parser.add_argument("--csv", type=str, default=None, help="Save metrics to CSV file")
    
    args = parser.parse_args()
    
    print(f"Extracting metrics from {args.log_dir}...")
    metrics = extract_metrics_from_logs(args.log_dir)
    print(f"Found {len(metrics)} metrics")
    
    print(f"Plotting metrics...")
    loss_metrics, perplexity_metrics = plot_metrics(metrics, args.output_dir, args.smooth)
    
    if args.csv:
        create_metrics_table(metrics, args.csv)

if __name__ == "__main__":
    main()