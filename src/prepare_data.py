"""
Prepare training data by creating metadata files and organizing preprocessed data.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

def organize_preprocessed_data(data_dir: str):
    """Create directories for preprocessed features.
    
    Args:
        data_dir: Path to processed data directory
    """
    data_dir = Path(data_dir)
    
    # Create directories for preprocessed features
    for split in ['train', 'val']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        for feature in ['text_tokens', 'mel', 'duration']:
            feature_dir = split_dir / feature
            feature_dir.mkdir(exist_ok=True, parents=True)
            
def prepare_metadata(data_dir: str):
    """Create metadata files for training and validation sets.
    
    Args:
        data_dir: Path to processed data directory
    """
    data_dir = Path(data_dir)
    
    # First create the feature directories
    organize_preprocessed_data(data_dir)
    
    # Initialize metadata lists
    train_metadata = []
    val_metadata = []
    
    # Process train and val directories
    for split in ['train', 'val']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        # Get all speaker directories
        speaker_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name in ['text_tokens', 'mel', 'duration']]
        for speaker_dir in tqdm(speaker_dirs, desc=f'Processing {split} data'):
            # Get speaker metadata
            meta_file = speaker_dir / 'metadata.json'
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Load preprocessed features
                tokens = torch.load(speaker_dir / 'tokens.pt')
                mel = torch.load(speaker_dir / 'mel.pt')
                duration = torch.load(speaker_dir / 'duration.pt')
                
                # Save as numpy files
                utterance_id = speaker_dir.name
                np.save(split_dir / 'text_tokens' / f'{utterance_id}.npy', tokens.numpy())
                np.save(split_dir / 'mel' / f'{utterance_id}.npy', mel.numpy())
                np.save(split_dir / 'duration' / f'{utterance_id}.npy', duration.numpy())
                
                # Create metadata entry
                entry = {
                    'id': utterance_id,
                    'text': metadata['text'],
                    'processed_text': metadata['processed_text'],
                    'speaker_id': metadata['speaker_id']
                }
                
                # Add to appropriate split
                if split == 'train':
                    train_metadata.append(entry)
                else:
                    val_metadata.append(entry)
    
    # Save metadata files
    if train_metadata:
        train_df = pd.DataFrame(train_metadata)
        train_df.to_csv(data_dir / 'train_metadata.csv', index=False)
        print(f"Train samples: {len(train_df)}")
        
    if val_metadata:
        val_df = pd.DataFrame(val_metadata)
        val_df.to_csv(data_dir / 'val_metadata.csv', index=False)
        print(f"Val samples: {len(val_df)}")

def main():
    # Prepare data
    data_dir = 'processed_data'
    print(f"Preparing data in {data_dir}")
    
    # Create metadata files and convert features
    prepare_metadata(data_dir)
    
    print("Data preparation complete!")

if __name__ == '__main__':
    main()
