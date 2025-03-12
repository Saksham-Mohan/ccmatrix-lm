#!/usr/bin/env python3
"""
Evaluation script for analyzing model performance based on linguistic structures.
This script evaluates the model's perplexity on different linguistic phenomena.
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import re

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import create_transformer_lm
from utils.logging import setup_logging
from utils.checkpointing import load_checkpoint

# Setup logging
logger = setup_logging(__name__)

LINGUISTIC_FEATURES = {
    'sentence_length': {
        'short': lambda s: len(s.split()) < 10,
        'medium': lambda s: 10 <= len(s.split()) < 20,
        'long': lambda s: len(s.split()) >= 20,
    },
    'syntactic_complexity': {
        'simple': lambda s: s.count(',') <= 1 and s.count(' and ') + s.count(' or ') <= 1,
        'complex': lambda s: s.count(',') > 1 or s.count(' and ') + s.count(' or ') > 1,
    },
    'rare_words': {
        'low': lambda s, vocab: calculate_rare_word_ratio(s, vocab) < 0.1,
        'medium': lambda s, vocab: 0.1 <= calculate_rare_word_ratio(s, vocab) < 0.2,
        'high': lambda s, vocab: calculate_rare_word_ratio(s, vocab) >= 0.2,
    },
    'morphological_richness': {
        'low': lambda s: calculate_morphological_score(s) < 0.3,
        'medium': lambda s: 0.3 <= calculate_morphological_score(s) < 0.5,
        'high': lambda s: calculate_morphological_score(s) >= 0.5,
    },
}

def calculate_rare_word_ratio(sentence, vocab, threshold=1000):
    """Calculate ratio of rare words in a sentence based on vocab rank."""
    words = sentence.lower().split()
    rare_count = sum(1 for word in words if word in vocab and vocab[word] > threshold)
    return rare_count / len(words) if words else 0

def calculate_morphological_score(sentence):
    """Calculate morphological richness score based on word endings."""
    words = sentence.lower().split()
    if not words:
        return 0
    
    suffix_patterns = [r'ing$', r'ed$', r'ly$', r'tion$', r'ment$', r'ness$', r'ity$', r'ous$', r'ive$', r'able$']
    suffix_count = sum(1 for word in words if any(re.search(pattern, word) for pattern in suffix_patterns))
    
    return suffix_count / len(words)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_perplexity(model, inputs, targets):
    """Compute perplexity for a batch of inputs."""
    logits = model(inputs, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets, logits
    )
    return tf.exp(tf.reduce_mean(loss))

def evaluate_by_linguistic_feature(model, tokenizer, test_file, vocab=None, batch_size=32, max_examples=None):
    """Evaluate model on test file, grouping by linguistic features."""
    logger.info(f"Evaluating model on {test_file}")
    
    if vocab is None:
        vocab = {}
        vocab_items = tokenizer.get_vocab().items()
        for word, idx in vocab_items:
            vocab[word] = idx
    
    with open(test_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    if max_examples and max_examples < len(sentences):
        sentences = sentences[:max_examples]

    categorized_sentences = {}
    for feature_name, categories in LINGUISTIC_FEATURES.items():
        categorized_sentences[feature_name] = {}
        for category_name, category_func in categories.items():
            if feature_name == 'rare_words':
                sentences_in_category = [s for s in sentences if category_func(s, vocab)]
            else:
                sentences_in_category = [s for s in sentences if category_func(s)]
            categorized_sentences[feature_name][category_name] = sentences_in_category
            logger.info(f"Found {len(sentences_in_category)} sentences in category {feature_name}/{category_name}")
    
    results = {}
    
    for feature_name, categories in categorized_sentences.items():
        results[feature_name] = {}
        
        for category_name, category_sentences in categories.items():
            if not category_sentences:
                logger.warning(f"No sentences in category {feature_name}/{category_name}, skipping")
                results[feature_name][category_name] = None
                continue
            
            all_input_ids = []
            for sentence in category_sentences:
                encoding = tokenizer.encode(sentence)
                all_input_ids.append(encoding.ids)
            
            num_batches = (len(all_input_ids) + batch_size - 1) // batch_size
            perplexity_sum = 0.0
            total_examples = 0
            
            for i in range(num_batches):
                batch_ids = all_input_ids[i * batch_size:(i + 1) * batch_size]
                
                if len(batch_ids) < 2:
                    continue
                
                max_length = max(len(ids) for ids in batch_ids)
                padded_ids = [ids + [0] * (max_length - len(ids)) for ids in batch_ids]
            
                inputs = tf.constant(padded_ids, dtype=tf.int32)
            
                batch_inputs = inputs[:, :-1]
                batch_targets = inputs[:, 1:]
                
                batch_perplexity = compute_perplexity(model, batch_inputs, batch_targets)
                
                perplexity_sum += batch_perplexity * len(batch_ids)
                total_examples += len(batch_ids)
            
            if total_examples > 0:
                avg_perplexity = perplexity_sum / total_examples
                results[feature_name][category_name] = float(avg_perplexity.numpy())
                logger.info(f"Category {feature_name}/{category_name}: Perplexity = {avg_perplexity:.4f}")
            else:
                results[feature_name][category_name] = None
                logger.warning(f"No valid batches for category {feature_name}/{category_name}")
    
    return results

def analyze_results(results, output_file=None):
    """Analyze evaluation results and output summary."""
    summary = {}
    
    for feature_name, categories in results.items():
        valid_categories = {k: v for k, v in categories.items() if v is not None}
        if not valid_categories:
            logger.warning(f"No valid results for feature {feature_name}")
            continue
        
        # Calculate statistics
        values = list(valid_categories.values())
        min_perp = min(values)
        max_perp = max(values)
        range_perp = max_perp - min_perp
        
        summary[feature_name] = {
            'categories': valid_categories,
            'min': min_perp,
            'max': max_perp,
            'range': range_perp,
            'relative_differences': {
                k: (v - min_perp) / min_perp if min_perp > 0 else 0
                for k, v in valid_categories.items()
            }
        }
        
        # Log summary
        logger.info(f"\nSummary for {feature_name}:")
        logger.info(f"  Min perplexity: {min_perp:.4f} ({min(valid_categories, key=valid_categories.get)})")
        logger.info(f"  Max perplexity: {max_perp:.4f} ({max(valid_categories, key=valid_categories.get)})")
        logger.info(f"  Range: {range_perp:.4f}")
        logger.info("  Relative differences:")
        for cat, rel_diff in summary[feature_name]['relative_differences'].items():
            logger.info(f"    {cat}: +{rel_diff*100:.2f}%")
    
    # Save results to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'raw_results': results,
                'summary': summary
            }, f, indent=2)
        logger.info(f"Saved analysis to {output_file}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate transformer language model on linguistic features")
    
    # Model and data arguments
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'en' for English)")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml", help="Path to model configuration file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--test_file", type=str, required=True, help="Test file to evaluate on")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, help="Path to save analysis results")
    
    args = parser.parse_args()
    
    model_config = load_config(args.model_config)
    
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    model = create_transformer_lm(
        vocab_size=vocab_size,
        d_model=model_config.get('d_model', 512),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 6),
        d_ff=model_config.get('d_ff', 2048),
        dropout_rate=model_config.get('dropout_rate', 0.1)
    )

    optimizer = tf.keras.optimizers.Adam()

    epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    vocab = {}
    for word, idx in tokenizer.get_vocab().items():
        vocab[word] = idx
    
    results = evaluate_by_linguistic_feature(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_file,
        vocab=vocab,
        batch_size=args.batch_size,
        max_examples=args.max_examples
    )
    
    output_file = args.output_file or f"results/{args.lang}_linguistic_analysis.json"
    analyze_results(results, output_file)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()