#!/usr/bin/env python3
"""
Script to download and prepare CCMatrix data for a specific language.
Usage: python download.py --lang en --output_dir data/raw
"""

import os
import argparse
import logging
import itertools
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_ccmatrix(lang, output_dir, max_sentences=None):
    """Download CCMatrix dataset for a specific language paired with English using streaming API."""
    logger.info("Downloading CCMatrix for language: " + lang)
    
    try:
     
        os.makedirs(output_dir, exist_ok=True)
        
        # For non-English languages, we download the lang-en pair
        # For English, we use en-fr as default (or allow specification of a different pair)
        pair = f"{lang}-en" if lang != "en" else "en-fr"
        
        logger.info("Loading dataset " + pair + " in streaming mode...")
        try:
            dataset = load_dataset("yhavinga/ccmatrix", pair, streaming=True)
        except ValueError as e:
            if "not a valid pair" in str(e).lower():
                # Try the reversed pair if the original isn't found
                reversed_pair = f"en-{lang}" if lang != "en" else "en-fr"
                logger.info("Pair " + pair + " not found, trying " + reversed_pair + " instead...")
                dataset = load_dataset("yhavinga/ccmatrix", reversed_pair, streaming=True)
                pair = reversed_pair
            else:
                raise
        
        output_file = os.path.join(output_dir, f"ccmatrix.{lang}.txt")
        
        logger.info("Streaming data to " + output_file + "...")
        
        stream = dataset['train']
        if max_sentences:
            stream = itertools.islice(stream, max_sentences)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # For progress tracking, we can't show a progress bar with total if we don't know the size
            for item in tqdm(stream, desc="Extracting " + lang + " sentences"):
                if lang == "en":
                    sentence = item['translation']['en']
                else:
                    # Handle both directions of the pair
                    if lang in item['translation']:
                        sentence = item['translation'][lang]
                    else:
                        # Fallback for when the language isn't directly available
                        other_lang = list(item['translation'].keys())
                        other_lang.remove('en') if 'en' in other_lang else None
                        if other_lang:
                            logger.warning("Language " + lang + " not found in translation, using " + other_lang[0] + " instead")
                            sentence = item['translation'][other_lang[0]]
                        else:
                            logger.warning("No suitable language found in translation, skipping")
                            continue
                
                f.write(sentence + '\n')
        
        logger.info("Extracted sentences to " + output_file)
        return output_file
        
    except ImportError:
        logger.error("datasets package not found. Please install with 'pip install datasets'")
        raise
    except Exception as e:
        logger.error("Error downloading dataset: " + str(e))
        raise

def extract_and_preprocess(input_file, output_file, max_sentences=None):
    """Extract and do basic preprocessing of the dataset."""
    logger.info("Preprocessing " + input_file + " to " + output_file)
    
    sentences_written = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(tqdm(infile, desc="Preprocessing")):
            if max_sentences and i >= max_sentences:
                break
                
            # Basic preprocessing: strip whitespace, filter empty lines
            line = line.strip()
            if line:
                outfile.write(line + '\n')
                sentences_written += 1
    
    logger.info("Wrote " + str(sentences_written) + " preprocessed sentences to " + output_file)
    return sentences_written

def main():
    parser = argparse.ArgumentParser(description="Download and prepare CCMatrix data")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'en' for English)")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory for raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--max_sentences", type=int, default=None, help="Maximum number of sentences to extract")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    
    raw_file = download_ccmatrix(args.lang, args.output_dir, args.max_sentences)

    processed_file = os.path.join(args.processed_dir, f"ccmatrix.{args.lang}.processed.txt")
    extract_and_preprocess(raw_file, processed_file, args.max_sentences)
    
    logger.info("Download and preprocessing complete!")

if __name__ == "__main__":
    main()