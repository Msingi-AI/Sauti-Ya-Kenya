"""
Data preprocessing for TTS training
"""
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import librosa
from src.preprocessor import TextPreprocessor

class DataPreprocessor:
    def __init__(self,
                 data_dir: str = "data",
                 output_dir: str = "processed_data",
                 sample_rate: int = 22050,
                 n_mel_channels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0,
                 train_split: float = 0.9):
        """
        Initialize data preprocessor
        Args:
            data_dir: Directory containing recordings and metadata
            output_dir: Directory to save processed data
            sample_rate: Target sample rate
            n_mel_channels: Number of mel channels
            mel_fmin: Minimum mel frequency
            mel_fmax: Maximum mel frequency
            train_split: Fraction of data to use for training
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.train_split = train_split
        
        # Create output directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.create_directories()
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mel_channels,
            f_min=mel_fmin,
            f_max=mel_fmax
        )
        
        # Initialize text preprocessor
        self.text_processor = TextPreprocessor()
        
    def create_directories(self):
        """Create necessary directories"""
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
    def load_metadata(self) -> Dict:
        """Load recording metadata"""
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            return json.load(f)
            
    def process_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Process audio file
        Args:
            audio_path: Path to audio file
        Returns:
            Processed audio tensor
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
            
        # Normalize
        audio = audio / torch.abs(audio).max()
        
        return audio
        
    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram
        Args:
            audio: Audio tensor
        Returns:
            Mel spectrogram tensor
        """
        # Add small offset to avoid log(0)
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
        
    def process_text(self, text: str) -> torch.Tensor:
        """
        Process text
        Args:
            text: Input text
        Returns:
            Token IDs tensor
        """
        tokens = self.text_processor.process_text(text)
        return torch.tensor(tokens.token_ids)
        
    def split_data(self, metadata: Dict) -> Tuple[List[str], List[str]]:
        """
        Split data into train and validation sets
        Args:
            metadata: Recording metadata
        Returns:
            Train and validation file lists
        """
        files = list(metadata.keys())
        random.shuffle(files)
        
        split_idx = int(len(files) * self.train_split)
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        return train_files, val_files
        
    def save_example(self,
                    audio: torch.Tensor,
                    mel: torch.Tensor,
                    tokens: torch.Tensor,
                    metadata: Dict,
                    filename: str,
                    output_dir: Path):
        """
        Save processed example
        Args:
            audio: Audio tensor
            mel: Mel spectrogram tensor
            tokens: Token IDs tensor
            metadata: Example metadata
            filename: Original filename
            output_dir: Output directory
        """
        example_dir = output_dir / filename.replace(".wav", "")
        example_dir.mkdir(exist_ok=True)
        
        # Save tensors
        torch.save(audio, example_dir / "audio.pt")
        torch.save(mel, example_dir / "mel.pt")
        torch.save(tokens, example_dir / "tokens.pt")
        
        # Save metadata
        with open(example_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def process_dataset(self):
        """Process entire dataset"""
        print("Loading metadata...")
        metadata = self.load_metadata()
        
        print("Splitting data...")
        train_files, val_files = self.split_data(metadata)
        
        print(f"Processing {len(train_files)} training examples...")
        for filename in tqdm(train_files):
            self.process_example(filename, metadata[filename], self.train_dir)
            
        print(f"Processing {len(val_files)} validation examples...")
        for filename in tqdm(val_files):
            self.process_example(filename, metadata[filename], self.val_dir)
            
    def process_example(self,
                       filename: str,
                       metadata: Dict,
                       output_dir: Path):
        """
        Process single example
        Args:
            filename: Audio filename
            metadata: Example metadata
            output_dir: Output directory
        """
        # Process audio
        audio_path = self.data_dir / "recordings" / filename
        audio = self.process_audio(audio_path)
        
        # Compute mel spectrogram
        mel = self.compute_mel_spectrogram(audio)
        
        # Process text
        tokens = self.process_text(metadata["text"])
        
        # Save processed example
        self.save_example(audio, mel, tokens, metadata, filename, output_dir)

def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()
