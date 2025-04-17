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

def download_ccmatrix(lang, pair, output_dir, max_sentences=None):
    """Download CCMatrix for a specific language pair and extract only the target language side."""
    logger.info(f"Downloading CCMatrix for language '{lang}' from pair '{pair}'")

    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Loading dataset '{pair}' in streaming mode...")
        try:
            dataset = load_dataset("yhavinga/ccmatrix", pair, streaming=True)
        except ValueError as e:
            logger.error(f"Dataset pair '{pair}' not found in CCMatrix: {str(e)}")
            raise

        output_file = os.path.join(output_dir, f"ccmatrix.{lang}.txt")

        logger.info(f"Streaming {lang} sentences to {output_file}...")

        stream = dataset['train']
        if max_sentences:
            stream = itertools.islice(stream, max_sentences)

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(stream, desc=f"Extracting '{lang}' sentences from '{pair}'"):
                translation = item.get("translation", {})
                sentence = translation.get(lang)

                if sentence:
                    f.write(sentence.strip() + '\n')
                else:
                    continue

        logger.info(f"✅ Finished writing {lang} sentences to: {output_file}")
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
    parser.add_argument("--lang", type=str, required=True, help="Target language to extract (e.g., 'en', 'ru')")
    parser.add_argument("--pair", type=str, required=True, help="Language pair from CCMatrix (e.g., 'en-ru', 'fr-en')")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory for raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--max_sentences", type=int, default=None, help="Maximum number of sentences to extract")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    raw_file = download_ccmatrix(args.lang, args.pair, args.output_dir, args.max_sentences)

    processed_file = os.path.join(args.processed_dir, f"ccmatrix.{args.lang}.processed.txt")
    extract_and_preprocess(raw_file, processed_file, args.max_sentences)

    logger.info("Download and preprocessing complete!")


if __name__ == "__main__":
    main()