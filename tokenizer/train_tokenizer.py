#!/usr/bin/env python3
"""
Script to train a BPE tokenizer on CCMatrix data.
Usage: python train_tokenizer.py --input_file data/processed/ccmatrix.en.processed.txt --vocab_size 32000
"""

import os
import argparse
import logging
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import NFKC
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_bpe_tokenizer(input_file, vocab_size, output_dir, max_tokens=None):
    """Train a BPE tokenizer on the provided text file."""
    logger.info(f"Training BPE tokenizer on {input_file} with vocab size {vocab_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    logger.info("Counting lines in the input file...")
    file_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    
    if max_tokens and max_tokens < file_lines:
        logger.info(f"Using {max_tokens} tokens for training")
        
        temp_file = os.path.join(output_dir, "temp_training_file.txt")
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(temp_file, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(tqdm(infile, total=file_lines, desc="Preparing training data")):
                if i >= max_tokens:
                    break
                outfile.write(line)
        
        training_files = [temp_file]
    else:
        training_files = [input_file]
    
    logger.info("Training tokenizer...")
    tokenizer.train(training_files, trainer)
    
    output_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(output_path)
    logger.info(f"Tokenizer saved to {output_path}")
    
    if max_tokens and max_tokens < file_lines:
        os.remove(temp_file)
        logger.info("Removed temporary training file")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on CCMatrix data")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file for training")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default="tokenizer/output", help="Output directory for the tokenizer")
    parser.add_argument("--max_tokens", type=int, default=None, help="Maximum number of tokens to use for training")
    
    args = parser.parse_args()
    
    train_bpe_tokenizer(args.input_file, args.vocab_size, args.output_dir, args.max_tokens)
    
    logger.info("Tokenizer training complete!")

if __name__ == "__main__":
    main()