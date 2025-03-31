"""
Extract and prepare text data from archive for tokenizer training.
"""
import os
import pandas as pd
from tqdm.auto import tqdm
import logging
import argparse

def extract_text_from_tsv(tsv_path: str) -> list[str]:
    """Extract text from TSV file."""
    df = pd.read_csv(tsv_path, sep='\t')
    if 'sentence' in df.columns:
        return df['sentence'].dropna().tolist()
    elif 'text' in df.columns:
        return df['text'].dropna().tolist()
    else:
        raise ValueError(f"No text column found in {tsv_path}")

def prepare_tokenizer_data(tsv_path: str, output_dir: str):
    """Extract text data and prepare for tokenizer training.
    
    Args:
        tsv_path: Path to the TSV file containing sentences
        output_dir: Directory to save extracted text data
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    
    # Extract text from TSV
    logger.info(f"Processing: {tsv_path}")
    try:
        texts = extract_text_from_tsv(tsv_path)
        logger.info(f"Extracted {len(texts)} texts")
        
        # Remove duplicates and empty lines
        texts = list(set(text.strip() for text in texts if text.strip()))
        logger.info(f"Total unique texts after cleaning: {len(texts)}")
        
        # Save texts to file
        output_file = os.path.join(text_dir, "tokenizer_training_data.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        logger.info(f"Saved text data to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing {tsv_path}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Prepare text data for tokenizer training")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to validated sentences TSV file")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save extracted data")
    
    args = parser.parse_args()
    prepare_tokenizer_data(args.input, args.output_dir)

if __name__ == '__main__':
    main()
