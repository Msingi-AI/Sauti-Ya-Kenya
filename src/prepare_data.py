import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

def organize_preprocessed_data(data_dir):
    data_dir = Path(data_dir)
    for split in ['train', 'val']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for feature in ['text_tokens', 'mel', 'duration']:
            feature_dir = split_dir / feature
            feature_dir.mkdir(exist_ok=True, parents=True)

def prepare_metadata(data_dir):
    data_dir = Path(data_dir)
    organize_preprocessed_data(data_dir)
    train_metadata = []
    val_metadata = []
    for split in ['train', 'val']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        speaker_dirs = [d for d in split_dir.iterdir() if d.is_dir() and not d.name in ['text_tokens', 'mel', 'duration']]
        for speaker_dir in tqdm(speaker_dirs, desc=f'Processing {split} data'):
            meta_file = speaker_dir / 'metadata.json'
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                tokens = torch.load(speaker_dir / 'tokens.pt')
                mel = torch.load(speaker_dir / 'mel.pt')
                duration = torch.load(speaker_dir / 'duration.pt')
                utterance_id = speaker_dir.name
                np.save(split_dir / 'text_tokens' / f'{utterance_id}.npy', tokens.numpy())
                np.save(split_dir / 'mel' / f'{utterance_id}.npy', mel.numpy())
                np.save(split_dir / 'duration' / f'{utterance_id}.npy', duration.numpy())
                entry = {
                    'id': utterance_id,
                    'text': metadata['text'],
                    'processed_text': metadata['processed_text'],
                    'speaker_id': metadata['speaker_id']
                }
                if split == 'train':
                    train_metadata.append(entry)
                else:
                    val_metadata.append(entry)
    if train_metadata:
        train_df = pd.DataFrame(train_metadata)
        train_df.to_csv(data_dir / 'train_metadata.csv', index=False)
        print(f"Train samples: {len(train_df)}")
    if val_metadata:
        val_df = pd.DataFrame(val_metadata)
        val_df.to_csv(data_dir / 'val_metadata.csv', index=False)
        print(f"Val samples: {len(val_df)}")

def main():
    data_dir = 'processed_data'
    print(f"Preparing data in {data_dir}")
    prepare_metadata(data_dir)
    print("Data preparation complete!")

if __name__ == '__main__':
    main()
