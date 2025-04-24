import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any

class TTSDataset(Dataset):
    def __init__(self, data_dir, metadata_file, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        speaker_id = row['speaker_id']
        clip_id = row['clip_id']
        text = row['text']
        mel_path = os.path.join(self.data_dir, speaker_id, f"{clip_id}_mel.pt")
        mel = torch.load(mel_path)
        text_ids = self.tokenizer.encode(text)
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        duration_path = os.path.join(self.data_dir, speaker_id, f"{clip_id}_duration.pt")
        duration = torch.load(duration_path) if os.path.exists(duration_path) else None
        sample = {
            'text_ids': text_ids,
            'mel_target': mel,
            'speaker_id': speaker_id,
            'duration': duration if duration is not None else torch.zeros_like(text_ids)
        }
        return sample
