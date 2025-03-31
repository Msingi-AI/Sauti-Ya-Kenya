"""
Training script for the Swahili tokenizer
"""
import argparse
import glob
import logging
import os
import sys
from typing import List
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import SwahiliTokenizer, TextPreprocessor
from src.config import ModelConfig

def load_text_files(data_dir: str, extensions: List[str] = ['.txt']) -> List[str]:
    """Load text data from files"""
    texts = []
    for ext in extensions:
        files = glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True)
        for file in tqdm(files, desc=f"Loading {ext} files"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    texts.extend(f.readlines())
            except Exception as e:
                logging.warning(f"Error loading {file}: {str(e)}")
    return [text.strip() for text in texts if text.strip()]

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    # Remove multiple spaces
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    text = ''.join(c for c in text if c.isalnum() or c.isspace() or c in '.,!?-')
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Train Swahili tokenizer")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing training text files")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=8000,
                       help="Vocabulary size for the tokenizer")
    parser.add_argument("--min-length", type=int, default=3,
                       help="Minimum text length to include")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess text data
    logging.info("Loading text data...")
    texts = load_text_files(args.data_dir)
    
    # Clean texts
    logging.info("Cleaning texts...")
    texts = [clean_text(text) for text in texts]
    texts = [text for text in texts if len(text) >= args.min_length]
    
    logging.info(f"Total texts for training: {len(texts)}")
    
    # Initialize and train tokenizer
    logging.info("Training tokenizer...")
    tokenizer = SwahiliTokenizer(vocab_size=args.vocab_size)
    model_path = os.path.join(args.output_dir, "tokenizer")
    tokenizer.train(texts, model_path)
    
    # Test tokenizer
    logging.info("Testing tokenizer...")
    test_texts = [
        "Habari yako! How are you doing leo?",
        "Niko sawa sana, asante.",
        "Tutaonana kesho asubuhi.",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        logging.info(f"\nInput: {text}")
        logging.info(f"Tokens: {tokens}")
        logging.info(f"Decoded: {decoded}")
    
    logging.info("Training complete! Tokenizer saved to: " + model_path + ".model")

if __name__ == "__main__":
    main()
