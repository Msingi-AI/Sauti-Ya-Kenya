"""
Extract and prepare text data from archive for tokenizer training.
"""
import os
import zipfile
import pandas as pd
from tqdm.auto import tqdm
import logging
import argparse

def extract_text_from_tsv(tsv_path: str) -> list[str]:
    """Extract text from TSV file."""
    df = pd.read_csv(tsv_path, sep='\t')
    if 'sentence' in df.columns:
        return df['sentence'].tolist()
    elif 'text' in df.columns:
        return df['text'].tolist()
    else:
        raise ValueError(f"No text column found in {tsv_path}")

def prepare_tokenizer_data(archive_path: str, output_dir: str):
    """Extract text data from archive and prepare for tokenizer training.
    
    Args:
        archive_path: Path to the zip archive containing dataset
        output_dir: Directory to save extracted text data
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    
    # Extract archive
    logger.info(f"Extracting archive: {archive_path}")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        # List all TSV files in archive
        tsv_files = [f for f in zip_ref.namelist() if f.endswith('.tsv')]
        
        # Process each TSV file
        all_texts = []
        for tsv_file in tsv_files:
            if 'validated' in tsv_file.lower():
                logger.info(f"Processing: {tsv_file}")
                # Extract TSV file
                zip_ref.extract(tsv_file, output_dir)
                tsv_path = os.path.join(output_dir, tsv_file)
                
                try:
                    # Extract text from TSV
                    texts = extract_text_from_tsv(tsv_path)
                    all_texts.extend(texts)
                    logger.info(f"Extracted {len(texts)} texts from {tsv_file}")
                except Exception as e:
                    logger.warning(f"Error processing {tsv_file}: {str(e)}")
                finally:
                    # Clean up extracted TSV
                    if os.path.exists(tsv_path):
                        os.remove(tsv_path)
    
    # Remove duplicates and empty lines
    all_texts = list(set(text.strip() for text in all_texts if text.strip()))
    logger.info(f"Total unique texts: {len(all_texts)}")
    
    # Save texts to file
    output_file = os.path.join(text_dir, "tokenizer_training_data.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')
    
    logger.info(f"Saved text data to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare text data for tokenizer training")
    parser.add_argument("--archive", type=str, required=True,
                       help="Path to dataset archive (zip)")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save extracted data")
    
    args = parser.parse_args()
    prepare_tokenizer_data(args.archive, args.output_dir)

if __name__ == '__main__':
    main()
