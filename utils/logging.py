#!/usr/bin/env python3
"""
Logging utilities for the training pipeline.
"""

import os
import logging
import sys
from datetime import datetime

def setup_logging(name, level=logging.INFO, log_file=None):
    """Setup logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_run_name(lang, prefix="run"):
    """Generate a run name based on language and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{lang}_{timestamp}"

class TensorboardWriter:
    """Wrapper class for TensorBoard logging."""
    
    def __init__(self, log_dir, enabled=True):
        self.enabled = enabled
        self.writer = None
        
        if enabled:
            import tensorflow as tf
            self.writer = tf.summary.create_file_writer(log_dir)
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        if self.enabled and self.writer:
            with self.writer.as_default():
                import tensorflow as tf
                tf.summary.scalar(tag, value, step=step)
    
    def log_text(self, tag, text, step):
        """Log text to TensorBoard."""
        if self.enabled and self.writer:
            with self.writer.as_default():
                import tensorflow as tf
                tf.summary.text(tag, text, step=step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram to TensorBoard."""
        if self.enabled and self.writer:
            with self.writer.as_default():
                import tensorflow as tf
                tf.summary.histogram(tag, values, step=step)
    
    def flush(self):
        """Flush the writer."""
        if self.enabled and self.writer:
            self.writer.flush()

class WandbLogger:
    """Wrapper class for Weights & Biases logging."""
    
    def __init__(self, project_name, run_name, config=None, enabled=True):
        self.enabled = enabled
        
        if enabled:
            try:
                import wandb
                
                # Initialize wandb
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config
                )
                self.wandb = wandb
            except ImportError:
                print("wandb not installed. Continuing without wandb logging.")
                self.enabled = False
    
    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        if self.enabled:
            self.wandb.log(metrics, step=step)
    
    def log_summary(self, metrics):
        """Log summary metrics to wandb."""
        if self.enabled:
            for key, value in metrics.items():
                self.wandb.run.summary[key] = value
    
    def finish(self):
        """Mark the run as finished."""
        if self.enabled:
            self.wandb.finish()