"""
Dataset module for TTS training
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any

class TTSDataset(Dataset):
    """Dataset for TTS training"""
    def __init__(self, data_dir: str, metadata_file: str, tokenizer):
        """
        Args:
            data_dir: Directory containing processed data
            metadata_file: Path to metadata CSV
            tokenizer: SwahiliTokenizer instance
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample"""
        row = self.metadata.iloc[idx]
        speaker_id = row['speaker_id']
        clip_id = row['clip_id']
        text = row['text']
        
        # Load mel spectrogram
        mel_path = os.path.join(self.data_dir, speaker_id, f"{clip_id}_mel.pt")
        mel = torch.load(mel_path)
        
        # Get text tokens
        text_ids = self.tokenizer.encode(text)
        text_ids = torch.tensor(text_ids, dtype=torch.long)
        
        # Load duration if available (for FastSpeech 2 training)
        duration_path = os.path.join(self.data_dir, speaker_id, f"{clip_id}_duration.pt")
        duration = torch.load(duration_path) if os.path.exists(duration_path) else None
        
        sample = {
            'text_ids': text_ids,
            'mel_target': mel,
            'speaker_id': speaker_id,
            'duration': duration if duration is not None else torch.zeros_like(text_ids)
        }
        
        return sample
