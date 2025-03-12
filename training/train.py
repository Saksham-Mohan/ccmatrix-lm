#!/usr/bin/env python3
"""
Main training script for the transformer language model.
"""

import os
import time
import json
import logging
import argparse
import tensorflow as tf
import yaml
import sys
from pathlib import Path

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import create_transformer_lm, PerplexityMetric
# Import only if needed
# from data.dataset import CCMatrixDataset
from utils.logging import setup_logging
from utils.checkpointing import save_checkpoint, load_checkpoint

# Setup logging
logger = setup_logging(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_tpu(tpu_name=None):
    """Setup TPU if available."""
    if tpu_name:
        logger.info(f"Connecting to TPU: {tpu_name}")
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            logger.info(f"TPU strategy initialized with {strategy.num_replicas_in_sync} replicas")
            return strategy
        except Exception as e:
            logger.error(f"Error setting up TPU: {e}")
            raise
    else:
        # Use GPU or CPU
        logger.info("Using default strategy (GPU or CPU)")
        return tf.distribute.get_strategy()

def compute_loss(labels, logits):
    """Compute cross entropy loss for language modeling."""
    # Use tf.nn implementation which is more consistent across TF versions
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
    )
    return tf.reduce_mean(loss)

def create_simple_lm(vocab_size, d_model=128):
    """Create a simple language model for testing."""
    inputs = tf.keras.layers.Input(shape=(None,))
    
    # Embedding layer
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    
    # Add a simple LSTM layer
    x = tf.keras.layers.LSTM(d_model, return_sequences=True)(x)
    
    # Output projection to vocabulary
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    logger.info(f"Created simple LSTM model with embedding dim {d_model}")
    
    return model

def train_step(model, optimizer, inputs, targets, loss_metric, perplexity_metric):
    """Training step for transformer language model with explicit debugging."""
    
    # Print shapes for debugging
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
    
    with tf.GradientTape() as tape:
        # Forward pass
        logits = model(inputs, training=True)
        print(f"Logits shape: {logits.shape}")
        
        # Compute loss - avoid using the compute_loss function for direct debugging
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits
        )
        loss = tf.reduce_mean(loss)
        
        print(f"Raw loss value: {loss.numpy()}")
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check gradients
    print(f"Number of gradient tensors: {len(gradients)}")
    
    # Check if any gradients are None or contain NaN/Inf
    valid_grads = True
    for i, g in enumerate(gradients):
        if g is None:
            print(f"Gradient {i} is None")
            valid_grads = False
        elif tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
            print(f"Gradient {i} contains NaN/Inf")
            valid_grads = False
    
    if valid_grads:
        # Print statistics of first gradient for debugging
        if gradients[0] is not None:
            print(f"First gradient stats: min={tf.reduce_min(gradients[0]).numpy()}, " 
                  f"max={tf.reduce_max(gradients[0]).numpy()}, " 
                  f"mean={tf.reduce_mean(gradients[0]).numpy()}")
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("Applied gradients successfully")
    else:
        print("Skipping gradient application due to invalid gradients")
    
    # Update metrics
    loss_metric.update_state(loss)
    
    # Calculate perplexity directly
    perplexity = tf.exp(loss)
    perplexity_metric.update_state(perplexity)
    
    print(f"Updated metrics - Loss: {loss_metric.result().numpy()}, Perplexity: {perplexity_metric.result().numpy()}")
    
    return loss

def validate(model, dataset, loss_metric, perplexity_metric):
    """Validate model on validation dataset."""
    for inputs, targets in dataset:
        logits = model(inputs, training=False)
        loss = compute_loss(targets, logits)
        
        # Update metrics
        loss_metric.update_state(loss)
        perplexity_metric.update_state(targets, logits)

def create_tf_dataset(encoded_file, batch_size, sequence_length, shuffle_buffer=None):
    """Create a TensorFlow dataset from encoded file."""
    # Calculate exact record size
    record_bytes = sequence_length * 4  # Each token is 4 bytes (int32)
    
    # Get file size and verify alignment
    file_size = os.path.getsize(encoded_file)
    if file_size % record_bytes != 0:
        logger.error(f"File size {file_size} is NOT a multiple of {record_bytes}.")
        raise ValueError("Dataset file is misaligned. Please re-encode the data.")
    
    num_examples = file_size // record_bytes
    logger.info(f"File size: {file_size} bytes, records: {num_examples}, record size: {record_bytes} bytes")
    
    # Create dataset from binary file
    dataset = tf.data.FixedLengthRecordDataset(
        encoded_file, 
        record_bytes=record_bytes
    )
    
    # Parse function for the dataset
    def parse_function(record):
        # Decode raw bytes to int32 tensor
        record = tf.io.decode_raw(record, tf.int32)
        
        # Input: all tokens except the last one
        inputs = record[:-1]
        # Target: all tokens except the first one (shifted by 1)
        targets = record[1:]
        
        return inputs, targets
    
    # Map, shuffle, batch
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle_buffer:
        dataset = dataset.shuffle(min(shuffle_buffer, num_examples))
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, num_examples

def train(args):
    """Main training function."""
    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    # Setup distribution strategy (TPU, GPU, or CPU)
    strategy = setup_tpu(args.tpu_name)
    
    # Initialize wandb if enabled
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    **model_config,
                    **training_config,
                    "lang": args.lang,
                    "max_tokens": args.max_tokens,
                },
                name=args.wandb_run_name or f"{args.lang}_{int(time.time())}"
            )
            wandb_available = True
        except ImportError:
            logger.warning("wandb not available, continuing without it")
            wandb_available = False
    else:
        wandb_available = False
    
    # Load tokenizer and get vocab size
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Load or create encoded data
    train_encoded_file = args.train_encoded_file or os.path.join(args.data_dir, f"ccmatrix.{args.lang}.train")
    val_encoded_file = args.val_encoded_file or os.path.join(args.data_dir, f"ccmatrix.{args.lang}.val")
    
    if not (os.path.exists(train_encoded_file) and os.path.exists(val_encoded_file)):
        logger.error(f"Encoded files not found: {train_encoded_file} and {val_encoded_file}")
        logger.error("Please run the dataset creation step first.")
        sys.exit(1)
    
    # Create TF datasets
    logger.info("Creating TensorFlow datasets...")
    
    # Create the datasets
    sequence_length = training_config.get('sequence_length', 512)
    batch_size = training_config.get('batch_size', 32)
    shuffle_buffer = training_config.get('shuffle_buffer', 10000)
    
    train_dataset, train_examples = create_tf_dataset(
        train_encoded_file,
        batch_size,
        sequence_length,
        shuffle_buffer
    )
    
    val_dataset, val_examples = create_tf_dataset(
        val_encoded_file,
        batch_size,
        sequence_length,
        shuffle_buffer=None
    )
    
    logger.info(f"Created training dataset with {train_examples} examples")
    logger.info(f"Created validation dataset with {val_examples} examples")
    
    # Calculate steps per epoch and validation steps
    steps_per_epoch = max(1, train_examples // training_config['batch_size'])
    validation_steps = max(1, val_examples // training_config['batch_size'])
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    
    # If we don't have enough validation data, adjust the batch size
    if val_examples < training_config['batch_size']:
        logger.warning(f"Not enough validation data for a full batch. Using smaller batch for validation.")
        val_batch_size = max(1, val_examples)
        val_dataset, val_examples = create_tf_dataset(
            val_encoded_file,
            val_batch_size,
            sequence_length,
            shuffle_buffer=None
        )
        validation_steps = 1
    
    # Prepare datasets for distribution strategy
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    # Model and optimizer
    with strategy.scope():
        # Create a simple model for testing instead of the complex transformer
        model = create_simple_lm(
            vocab_size=vocab_size,
            d_model=128  # Use a smaller dimension for faster training
        )
        
        # Use a fixed learning rate for testing
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_perplexity = tf.keras.metrics.Mean(name='train_perplexity')
        
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_perplexity = tf.keras.metrics.Mean(name='val_perplexity')
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        logger.info(f"Loaded checkpoint from {args.checkpoint_path}, starting from epoch {start_epoch}")
    
    # Define distributed train step - remove @tf.function decorator for debugging
    # @tf.function
    def distributed_train_step(dist_inputs, dist_targets):
        per_replica_losses = strategy.run(
            train_step, args=(model, optimizer, dist_inputs, dist_targets, train_loss, train_perplexity)
        )
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    
    # Define distributed validation step - remove @tf.function decorator for debugging
    # @tf.function
    def distributed_val_step(dist_inputs, dist_targets):
        strategy.run(
            lambda inputs, targets: validate(model, [(inputs, targets)], val_loss, val_perplexity),
            args=(dist_inputs, dist_targets)
        )
    
    # Setup TensorBoard if enabled
    if args.tensorboard:
        tb_dir = os.path.join(args.log_dir, f"{args.lang}_{int(time.time())}")
        summary_writer = tf.summary.create_file_writer(tb_dir)
        logger.info(f"TensorBoard logs will be saved to {tb_dir}")
    
    # Create checkpoint directory if it doesn't exist
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info(f"Starting training for {training_config['epochs']} epochs")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, training_config['epochs']):
        start_time = time.time()
        
        # Reset metrics for this epoch
        train_loss.reset_state()
        train_perplexity.reset_state()
        val_loss.reset_state()
        val_perplexity.reset_state()
        
        # Training
        for step, (inputs, targets) in enumerate(train_dist_dataset):
            loss = distributed_train_step(inputs, targets)
            
            if step % 100 == 0:
                current_loss = train_loss.result()
                current_perplexity = train_perplexity.result()
                logger.info(f"Epoch {epoch+1}, Step {step}, Loss: {current_loss:.4f}, Perplexity: {current_perplexity:.4f}")
                
                # Log to wandb if enabled
                if wandb_available:
                    wandb.log({
                        "train_loss": current_loss,
                        "train_perplexity": current_perplexity,
                        "epoch": epoch + 1,
                        "step": step
                    })
        
        # Validation
        if validation_steps > 0:
            for inputs, targets in val_dist_dataset:
                distributed_val_step(inputs, targets)
                
            # Compute validation results
            val_loss_result = val_loss.result()
            val_perplexity_result = val_perplexity.result()
        else:
            # Skip validation if no validation steps
            val_loss_result = 0.0
            val_perplexity_result = 0.0
            logger.warning("Skipping validation due to insufficient validation data.")
        
        # Compute epoch results
        train_loss_result = train_loss.result().numpy()
        train_perplexity_result = train_perplexity.result().numpy()
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        logger.info(f"  Train Loss: {train_loss_result:.4f}, Train Perplexity: {train_perplexity_result:.4f}")
        logger.info(f"  Val Loss: {val_loss_result:.4f}, Val Perplexity: {val_perplexity_result:.4f}")
        
        # Log to wandb if enabled
        if wandb_available:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss_epoch": train_loss_result,
                "train_perplexity_epoch": train_perplexity_result,
                "val_loss": val_loss_result,
                "val_perplexity": val_perplexity_result,
                "epoch_time": epoch_time
            })
        
        # Log to TensorBoard if enabled
        if args.tensorboard:
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss_result, step=epoch+1)
                tf.summary.scalar('train_perplexity', train_perplexity_result, step=epoch+1)
                tf.summary.scalar('val_loss', val_loss_result, step=epoch+1)
                tf.summary.scalar('val_perplexity', val_perplexity_result, step=epoch+1)
        
        # Save checkpoint
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss_result < best_val_loss:
            best_val_loss = val_loss_result
            if args.checkpoint_dir:
                best_model_path = os.path.join(args.checkpoint_dir, "best_model")
                save_checkpoint(model, optimizer, epoch+1, best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        
    logger.info("Training completed!")
    
    # Log final metrics to wandb if enabled
    if wandb_available:
        wandb.log({
            "final_val_loss": val_loss_result,
            "final_val_perplexity": val_perplexity_result,
            "best_val_loss": best_val_loss
        })
        wandb.finish()
    
    return model

def train_with_keras_fit(args):
    """Train using Keras model.fit() API instead of custom training loop."""
    # Load configurations
    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    # Load tokenizer and get vocab size
    tokenizer = load_tokenizer(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Load or create encoded data
    train_encoded_file = args.train_encoded_file or os.path.join(args.data_dir, f"ccmatrix.{args.lang}.train")
    val_encoded_file = args.val_encoded_file or os.path.join(args.data_dir, f"ccmatrix.{args.lang}.val")
    
    if not (os.path.exists(train_encoded_file) and os.path.exists(val_encoded_file)):
        logger.error(f"Encoded files not found: {train_encoded_file} and {val_encoded_file}")
        logger.error("Please run the dataset creation step first.")
        sys.exit(1)
    
    # Create TF datasets
    logger.info("Creating TensorFlow datasets...")
    sequence_length = training_config.get('sequence_length', 512)
    batch_size = training_config.get('batch_size', 32)
    
    # Create repeatable datasets
    def create_repeatable_dataset(encoded_file, batch_size, sequence_length, shuffle_buffer=None):
        """Create a TensorFlow dataset that can be used for multiple epochs."""
        # Get base dataset
        dataset, num_examples = create_tf_dataset(
            encoded_file,
            batch_size,
            sequence_length,
            shuffle_buffer
        )
        
        # Add repeat() to make the dataset usable for multiple epochs
        dataset = dataset.repeat()
        
        return dataset, num_examples
    
    # Create datasets with repeat for multiple epochs
    train_dataset, train_examples = create_repeatable_dataset(
        train_encoded_file,
        batch_size,
        sequence_length,
        shuffle_buffer=1000
    )
    
    val_dataset, val_examples = create_repeatable_dataset(
        val_encoded_file,
        batch_size,
        sequence_length,
        shuffle_buffer=None
    )
    
    # Calculate steps per epoch
    steps_per_epoch = train_examples // batch_size
    validation_steps = max(1, val_examples // batch_size)
    
    logger.info(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
    
    # Create a simpler model for testing
    model = create_simple_lm(vocab_size, d_model=128)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy']
    )
    
    # Create callbacks
    callbacks = []
    
    # Add TensorBoard callback if enabled
    if args.tensorboard:
        tb_dir = os.path.join(args.log_dir, f"{args.lang}_fit_{int(time.time())}")
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tb_callback)
        logger.info(f"TensorBoard logs will be saved to {tb_dir}")
    
    # Add model checkpoint callback
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, "model_{epoch:02d}.weights.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=False
        )
        callbacks.append(checkpoint_callback)
    
    # Train the model
    logger.info("Starting training with model.fit()...")
    model.fit(
        train_dataset,
        epochs=training_config.get('epochs', 20),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    logger.info("Training completed!")
    return model

def load_tokenizer(tokenizer_path):
    """Load tokenizer from file."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(tokenizer_path)

def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model on CCMatrix data")
    
    # Data arguments
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'en' for English)")
    parser.add_argument("--input_file", type=str, help="Processed text file for training")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory for processed data")
    parser.add_argument("--train_encoded_file", type=str, help="Pre-encoded training file")
    parser.add_argument("--val_encoded_file", type=str, help="Pre-encoded validation file")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to process")
    
    # Model arguments
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml", help="Path to model configuration file")
    parser.add_argument("--training_config", type=str, default="config/training_config.yaml", help="Path to training configuration file")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_path", type=str, help="Path to load a checkpoint from")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="ccmatrix-lm", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")
    
    # TPU arguments
    parser.add_argument("--tpu_name", type=str, help="TPU name for training")
    
    # Use alternative training approach
    parser.add_argument("--use_fit", action="store_true", help="Use model.fit() instead of custom training loop")
    
    args = parser.parse_args()
    
    # Check for input file
    if not args.input_file and not (args.train_encoded_file and args.val_encoded_file):
        args.input_file = os.path.join(args.data_dir, f"ccmatrix.{args.lang}.processed.txt")
        if not os.path.exists(args.input_file):
            parser.error(f"Input file not found: {args.input_file}. Please provide --input_file or --train_encoded_file and --val_encoded_file.")
    
    # Train the model using the selected approach
    if args.use_fit:
        train_with_keras_fit(args)
    else:
        train(args)

if __name__ == "__main__":
    main()