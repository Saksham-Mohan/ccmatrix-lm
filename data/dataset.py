#!/usr/bin/env python3
"""
Script to create TensorFlow datasets from tokenized CCMatrix data.
"""

import os
import logging
import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm
import struct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CCMatrixDataset:
    def __init__(self, tokenizer_path, sequence_length=512):
        """Initialize dataset with tokenizer and sequence length."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.sequence_length = sequence_length
        logger.info(f"Initialized CCMatrixDataset with sequence length {sequence_length}")
        
    def encode_text(self, text_file, output_file, max_examples=None):
        """Encode text file into tokenized format."""
        logger.info(f"Encoding text from {text_file} to {output_file}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Count lines for progress bar
        file_lines = sum(1 for _ in open(text_file, 'r', encoding='utf-8'))
        total_lines = min(file_lines, max_examples) if max_examples else file_lines
        
        # Process and encode text
        with open(text_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'wb') as f_out:
            
            tokens_total = 0
            examples_processed = 0
            current_ids = []
            
            for line in tqdm(f_in, total=total_lines, desc="Encoding text"):
                if max_examples and examples_processed >= max_examples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    encoded = self.tokenizer.encode(line)
                    token_ids = encoded.ids
                    
                    current_ids.extend(token_ids)  # Extend with tokenized IDs
                    
                    # Write sequences of exact `sequence_length`
                    while len(current_ids) >= self.sequence_length:
                        sequence = current_ids[:self.sequence_length]
                        current_ids = current_ids[self.sequence_length:]
                        
                        # Write sequence as int32 values
                        f_out.write(struct.pack(f'{self.sequence_length}i', *sequence))
                        
                        tokens_total += self.sequence_length
                        examples_processed += 1
                        
                        if max_examples and examples_processed >= max_examples:
                            break
                except Exception as e:
                    logger.warning(f"Error processing line: {e}")
                    continue
            
            # **Ensure last batch is properly padded to `sequence_length`**
            if current_ids:
                if len(current_ids) < self.sequence_length:
                    padding = [0] * (self.sequence_length - len(current_ids))
                    current_ids.extend(padding)  # Dynamically pad to sequence_length

                if len(current_ids) > self.sequence_length:
                    logger.error(f"Error: Sequence length exceeded expected {self.sequence_length}. Got {len(current_ids)}")
                
                f_out.write(struct.pack(f'{self.sequence_length}i', *current_ids[:self.sequence_length]))  # Trim to ensure correct size
                tokens_total += self.sequence_length
                examples_processed += 1
        
        logger.info(f"Encoded {examples_processed} examples with {tokens_total} tokens total")
        return examples_processed
    
    def create_tf_dataset(self, encoded_file, batch_size, shuffle_buffer=10000):
        """Create a TensorFlow dataset from encoded file."""
        logger.info(f"Creating TF dataset from {encoded_file} with batch size {batch_size}")
        
        # **Compute dynamic record size (NO hardcoding)**
        record_bytes = self.sequence_length * 4  # Each int32 is 4 bytes
        logger.info(f"Reading records with record size: {record_bytes} bytes")
        
        # **Verify file alignment before reading**
        file_size = os.path.getsize(encoded_file)
        if file_size % record_bytes != 0:
            logger.error(f"File size {file_size} is NOT a multiple of {record_bytes}.")
            raise ValueError("Dataset file is misaligned. Please re-encode the data.")
        else:
            logger.info(f"Dataset file size is correct: {file_size} bytes (multiple of {record_bytes}).")

        try:
            dataset = tf.data.FixedLengthRecordDataset(
                encoded_file, 
                record_bytes=record_bytes
            )
            
            def parse_function(record):
                record = tf.io.decode_raw(record, tf.int32)
                inputs = record[:-1]
                targets = record[1:]
                return inputs, targets
            
            dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            
            if shuffle_buffer:
                dataset = dataset.shuffle(min(shuffle_buffer, file_size // record_bytes))
                
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset, file_size // record_bytes
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise
    
    @staticmethod
    def split_train_val(encoded_file, train_ratio=0.95, sequence_length=512):
        """Split encoded file into train and validation files."""
        logger.info(f"Splitting {encoded_file} into train and validation sets")
        
        base_path = os.path.splitext(encoded_file)[0]
        train_file = f"{base_path}.train"
        val_file = f"{base_path}.val"
        
        # Calculate record size in bytes
        record_bytes = sequence_length * 4  # Each int32 token is 4 bytes
        
        # Get file size and calculate number of records
        file_size = os.path.getsize(encoded_file)
        total_records = file_size // record_bytes
        
        # Calculate how many records should go to training
        train_records = int(total_records * train_ratio)
        train_size = train_records * record_bytes
        
        logger.info(f"File size: {file_size} bytes, total records: {total_records}")
        logger.info(f"Train records: {train_records}, train size: {train_size} bytes")
        
        with open(encoded_file, 'rb') as f_in, \
            open(train_file, 'wb') as f_train, \
            open(val_file, 'wb') as f_val:
            
            # Copy exact number of complete records to train file
            bytes_copied = 0
            buffer_size = min(1024 * 1024, train_size)  # 1MB buffer or smaller
            
            while bytes_copied < train_size:
                chunk_size = min(buffer_size, train_size - bytes_copied)
                f_train.write(f_in.read(chunk_size))
                bytes_copied += chunk_size
            
            # Copy remaining records to validation file
            while True:
                chunk = f_in.read(buffer_size)
                if not chunk:
                    break
                f_val.write(chunk)
        
        # Verify file sizes
        train_file_size = os.path.getsize(train_file)
        val_file_size = os.path.getsize(val_file)
        
        logger.info(f"Created train file {train_file} ({train_file_size} bytes, {train_file_size//record_bytes} records)")
        logger.info(f"Created validation file {val_file} ({val_file_size} bytes, {val_file_size//record_bytes} records)")
        
        # Verify alignment
        if train_file_size % record_bytes != 0:
            logger.error(f"Train file size {train_file_size} is not a multiple of record size {record_bytes}")
        if val_file_size % record_bytes != 0:
            logger.error(f"Validation file size {val_file_size} is not a multiple of record size {record_bytes}")
        
        return train_file, val_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create TensorFlow datasets from CCMatrix data")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file")
    parser.add_argument("--input_file", type=str, required=True, help="Input text file")
    parser.add_argument("--output_file", type=str, required=True, help="Output encoded file")
    parser.add_argument("--sequence_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to encode")
    parser.add_argument("--split", action="store_true", help="Split into train and validation sets")
    
    args = parser.parse_args()
    
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    dataset = CCMatrixDataset(args.tokenizer_path, args.sequence_length)
    dataset.encode_text(args.input_file, args.output_file, args.max_examples)
    
    if args.split:
        dataset.split_train_val(args.output_file, sequence_length=args.sequence_length)
    
    logger.info("Dataset creation complete!")

if __name__ == "__main__":
    main()

