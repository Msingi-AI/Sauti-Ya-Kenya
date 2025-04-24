import os
import pandas as pd
from tqdm.auto import tqdm
import logging
import argparse

def extract_text_from_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    if 'sentence' in df.columns:
        return df['sentence'].dropna().tolist()
    elif 'text' in df.columns:
        return df['text'].dropna().tolist()
    else:
        raise ValueError(f"No text column found in {tsv_path}")

def prepare_tokenizer_data(tsv_path, output_dir):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    logger.info(f"Processing: {tsv_path}")
    try:
        texts = extract_text_from_tsv(tsv_path)
        logger.info(f"Extracted {len(texts)} texts")
        texts = list(set(text.strip() for text in texts if text.strip()))
        logger.info(f"Total unique texts after cleaning: {len(texts)}")
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
    parser.add_argument("--input", type=str, required=True, help="Path to validated sentences TSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted data")
    args = parser.parse_args()
    prepare_tokenizer_data(args.input, args.output_dir)

if __name__ == '__main__':
    main()
