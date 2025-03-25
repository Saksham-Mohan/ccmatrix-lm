#!/usr/bin/env python3
"""
Script to parse training logs and generate visualizations of model metrics.
"""

import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_metrics_from_log(log_file):
    """Extract training and validation metrics from log file."""
    logger.info(f"Reading log file: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_text = f.read()
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return None, None
    
    # Extract epoch summaries
    epoch_pattern = r"Epoch (\d+) completed.*?\n.*?Train Loss: ([\d\.]+), Train Perplexity: ([\d\.]+).*?\n.*?Val Loss: ([\d\.]+), Val Perplexity: ([\d\.]+)"
    epoch_matches = re.findall(epoch_pattern, log_text)
    
    # Extract step metrics (within epochs)
    step_pattern = r"Epoch (\d+), Step (\d+), Loss: ([\d\.]+), Perplexity: ([\d\.]+)"
    step_matches = re.findall(step_pattern, log_text)
    
    logger.info(f"Found {len(epoch_matches)} epoch summaries and {len(step_matches)} step updates")
    
    if not epoch_matches:
        logger.warning("No epoch data found. Check log format.")
        # Try alternative pattern without validation data
        alt_epoch_pattern = r"Epoch (\d+) completed.*?\n.*?Train Loss: ([\d\.]+), Train Perplexity: ([\d\.]+)"
        alt_epoch_matches = re.findall(alt_epoch_pattern, log_text)
        if alt_epoch_matches:
            logger.info(f"Found {len(alt_epoch_matches)} epoch summaries with training data only")
            # Process these differently
            metrics = defaultdict(list)
            for match in alt_epoch_matches:
                epoch = int(match[0])
                metrics['epoch'].append(epoch)
                metrics['train_loss'].append(float(match[1]))
                metrics['train_perplexity'].append(float(match[2]))
            return metrics, None
        
    # Organize the epoch data
    metrics = defaultdict(list)
    
    for match in epoch_matches:
        epoch = int(match[0])
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(float(match[1]))
        metrics['train_perplexity'].append(float(match[2]))
        metrics['val_loss'].append(float(match[3]))
        metrics['val_perplexity'].append(float(match[4]))
    
    # Print ranges to help with debugging
    if metrics['train_perplexity']:
        logger.info(f"Train perplexity range: {min(metrics['train_perplexity'])} to {max(metrics['train_perplexity'])}")
    if 'val_perplexity' in metrics and metrics['val_perplexity']:
        logger.info(f"Val perplexity range: {min(metrics['val_perplexity'])} to {max(metrics['val_perplexity'])}")
    
    # Create step-level metrics
    step_metrics = defaultdict(list)
    for match in step_matches:
        epoch = int(match[0])
        step = int(match[1])
        loss = float(match[2])
        perplexity = float(match[3])
        
        # Use a combined step number for continuous plotting
        global_step = (epoch - 1) * 100 + step  # Assuming max 100 steps per epoch
        
        step_metrics['global_step'].append(global_step)
        step_metrics['epoch'].append(epoch)
        step_metrics['step'].append(step)
        step_metrics['loss'].append(loss)
        step_metrics['perplexity'].append(perplexity)
    
    return metrics, step_metrics

def plot_training_metrics(metrics, step_metrics, output_dir):
    """Create plots for loss and perplexity trends."""
    os.makedirs(output_dir, exist_ok=True)
    
    # If no metrics found, return early
    if not metrics or not metrics['epoch']:
        logger.warning("No metrics to plot")
        return
    
    # PLOT 1: Epoch-level metrics (Linear scale)
    plt.figure(figsize=(15, 10))
    
    # Plot 1.1: Loss by epoch
    plt.subplot(2, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], 'b-o', label='Train Loss')
    if 'val_loss' in metrics and metrics['val_loss']:
        plt.plot(metrics['epoch'], metrics['val_loss'], 'r-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.grid(True)
    plt.legend()
    
    # Plot 1.2: Perplexity by epoch (linear scale)
    plt.subplot(2, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_perplexity'], 'b-o', label='Train Perplexity')
    if 'val_perplexity' in metrics and metrics['val_perplexity']:
        plt.plot(metrics['epoch'], metrics['val_perplexity'], 'r-o', label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Epoch')
    plt.grid(True)
    plt.legend()
    
    # Plot 1.3: Perplexity by epoch (log scale)
    plt.subplot(2, 2, 3)
    plt.semilogy(metrics['epoch'], metrics['train_perplexity'], 'b-o', label='Train Perplexity')
    if 'val_perplexity' in metrics and metrics['val_perplexity']:
        plt.semilogy(metrics['epoch'], metrics['val_perplexity'], 'r-o', label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (log scale)')
    plt.title('Perplexity vs. Epoch (Log Scale)')
    plt.grid(True)
    plt.legend()
    
    # Plot 1.4: Loss ratio (val/train) if validation metrics exist
    if 'val_loss' in metrics and metrics['val_loss']:
        plt.subplot(2, 2, 4)
        loss_ratio = [v/t for v, t in zip(metrics['val_loss'], metrics['train_loss'])]
        plt.plot(metrics['epoch'], loss_ratio, 'g-o', label='Val/Train Loss Ratio')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Ratio')
        plt.title('Validation/Training Loss Ratio')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epoch_metrics.png'), dpi=300)
    plt.close()
    
    # PLOT 2: Step-level metrics if available
    if step_metrics and step_metrics['global_step']:
        plt.figure(figsize=(15, 10))
        
        # Plot 2.1: Loss by step
        plt.subplot(2, 1, 1)
        plt.plot(step_metrics['global_step'], step_metrics['loss'], 'g-', label='Training Loss')
        # Mark epoch boundaries
        epoch_boundaries = []
        prev_epoch = None
        for i, epoch in enumerate(step_metrics['epoch']):
            if epoch != prev_epoch:
                epoch_boundaries.append(step_metrics['global_step'][i])
                prev_epoch = epoch
        
        for boundary in epoch_boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Loss vs. Training Step')
        plt.grid(True)
        plt.legend()
        
        # Plot 2.2: Perplexity by step
        plt.subplot(2, 1, 2)
        plt.plot(step_metrics['global_step'], step_metrics['perplexity'], 'b-', label='Training Perplexity')
        # Mark epoch boundaries
        for boundary in epoch_boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('Training Step')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs. Training Step')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step_metrics.png'), dpi=300)
        plt.close()
        
        # PLOT 3: Step-level perplexity (log scale)
        plt.figure(figsize=(10, 6))
        plt.semilogy(step_metrics['global_step'], step_metrics['perplexity'], 'b-', label='Training Perplexity')
        # Mark epoch boundaries
        for boundary in epoch_boundaries:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.3)
        
        plt.xlabel('Training Step')
        plt.ylabel('Perplexity (log scale)')
        plt.title('Perplexity vs. Training Step (Log Scale)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step_perplexity_log.png'), dpi=300)
        plt.close()
    
    logger.info(f"Plots saved to {output_dir}")

def create_metrics_csv(metrics, step_metrics, output_file):
    """Save metrics to a CSV file."""
    # Create epoch-level DataFrame
    epoch_df = pd.DataFrame({
        'epoch': metrics['epoch'],
        'train_loss': metrics['train_loss'],
        'train_perplexity': metrics['train_perplexity'],
    })
    
    if 'val_loss' in metrics and metrics['val_loss']:
        epoch_df['val_loss'] = metrics['val_loss']
        epoch_df['val_perplexity'] = metrics['val_perplexity']
    
    # Create step-level DataFrame if available
    if step_metrics and step_metrics['global_step']:
        step_df = pd.DataFrame({
            'global_step': step_metrics['global_step'],
            'epoch': step_metrics['epoch'],
            'step': step_metrics['step'],
            'loss': step_metrics['loss'],
            'perplexity': step_metrics['perplexity']
        })
        
        # Save both to CSV
        epoch_df.to_csv(output_file, index=False)
        step_df.to_csv(output_file.replace('.csv', '_steps.csv'), index=False)
        
        logger.info(f"Metrics saved to {output_file} and {output_file.replace('.csv', '_steps.csv')}")
    else:
        # Save just epoch metrics
        epoch_df.to_csv(output_file, index=False)
        logger.info(f"Metrics saved to {output_file}")

def print_training_summary(metrics):
    """Print a summary of the training progress."""
    if not metrics or not metrics['epoch']:
        logger.warning("No metrics available for summary")
        return
    
    logger.info("\nTraining Summary:")
    logger.info(f"Completed {max(metrics['epoch'])} epochs")
    logger.info(f"Initial train loss: {metrics['train_loss'][0]:.4f}, Initial train perplexity: {metrics['train_perplexity'][0]:.2f}")
    logger.info(f"Final train loss: {metrics['train_loss'][-1]:.4f}, Final train perplexity: {metrics['train_perplexity'][-1]:.2f}")
    
    if 'val_loss' in metrics and metrics['val_loss']:
        logger.info(f"Initial val loss: {metrics['val_loss'][0]:.4f}, Initial val perplexity: {metrics['val_perplexity'][0]:.2f}")
        logger.info(f"Final val loss: {metrics['val_loss'][-1]:.4f}, Final val perplexity: {metrics['val_perplexity'][-1]:.2f}")
    
    train_loss_reduction = (metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100
    logger.info(f"Train loss reduced by {train_loss_reduction:.2f}%")
    
    if 'val_loss' in metrics and metrics['val_loss']:
        val_loss_reduction = (metrics['val_loss'][0] - metrics['val_loss'][-1]) / metrics['val_loss'][0] * 100
        logger.info(f"Validation loss reduced by {val_loss_reduction:.2f}%")
    
    logger.info(f"Perplexity improvement factor: {metrics['train_perplexity'][0] / metrics['train_perplexity'][-1]:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Parse and plot training metrics from log files")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the training log file")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    parser.add_argument("--csv", type=str, default=None, help="Path to save CSV metrics file")
    
    args = parser.parse_args()
    
    metrics, step_metrics = extract_metrics_from_log(args.log_file)
    
    if metrics is None:
        logger.error("Failed to extract metrics from log file")
        return
    
    if not metrics['epoch']:
        logger.warning("No epoch metrics found in the log file.")
        return
    
    plot_training_metrics(metrics, step_metrics, args.output_dir)
    
    if args.csv:
        create_metrics_csv(metrics, step_metrics, args.csv)
    
    print_training_summary(metrics)

if __name__ == "__main__":
    main()