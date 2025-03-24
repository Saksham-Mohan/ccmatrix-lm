#!/usr/bin/env python3
"""
Utility functions for model checkpointing.
"""

import os
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save model and optimizer checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        model_path = f"{checkpoint_path}_model.keras"
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")
        
        try:
            optimizer_path = f"{checkpoint_path}_optimizer.npy"
            optimizer_weights = optimizer.get_weights()
            np.save(optimizer_path, optimizer_weights)
            logger.info(f"Saved optimizer state to {optimizer_path}")
        except Exception as e:
            logger.warning(f"Could not save optimizer state: {e}")
        
        epoch_path = f"{checkpoint_path}_epoch.txt"
        with open(epoch_path, 'w') as f:
            f.write(str(epoch))
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
    
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer from checkpoint."""
    try:
        # Load model from saved model
        model_path = f"{checkpoint_path}_model.keras"
        if os.path.exists(model_path):
            model.load_weights(model_path)
            logger.info(f"Loaded model weights from {model_path}")
        else:
            # Try the old path format as fallback
            legacy_model_path = f"{checkpoint_path}_model"
            if os.path.exists(legacy_model_path):
                model.load_weights(legacy_model_path)
                logger.info(f"Loaded model weights from {legacy_model_path}")
            else:
                logger.warning(f"No model weights found at {model_path} or {legacy_model_path}")
        
        # Load optimizer state if it exists
        optimizer_path = f"{checkpoint_path}_optimizer.npy"
        if os.path.exists(optimizer_path):
            optimizer_weights = np.load(optimizer_path, allow_pickle=True)
            optimizer.set_weights(optimizer_weights)
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        
        # Load epoch info
        epoch_path = f"{checkpoint_path}_epoch.txt"
        if os.path.exists(epoch_path):
            with open(epoch_path, 'r') as f:
                epoch = int(f.read().strip())
            logger.info(f"Loaded checkpoint from epoch {epoch}")
            return epoch
        
        return 0
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0

def list_checkpoints(checkpoint_dir):
    """List available checkpoints in the directory."""
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith("_info.json"):
            checkpoint_name = filename.replace("_info.json", "")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # Load epoch info
            with open(os.path.join(checkpoint_dir, filename), 'r') as f:
                info = json.load(f)
            
            checkpoints.append({
                'name': checkpoint_name,
                'path': checkpoint_path,
                'epoch': info.get('epoch', 0)
            })
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'])
    
    return checkpoints

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint in the directory."""
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        logger.warning(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    # Return the checkpoint with the highest epoch
    latest_checkpoint = max(checkpoints, key=lambda x: x['epoch'])
    logger.info(f"Found latest checkpoint: {latest_checkpoint['name']} (epoch {latest_checkpoint['epoch']})")
    
    return latest_checkpoint['path']