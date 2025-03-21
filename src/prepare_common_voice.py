"""
Prepare Mozilla Common Voice Swahili dataset for TTS training.
"""
import os
import torch
import torchaudio
import pandas as pd
from typing import Dict, List, Tuple
from datasets import load_dataset, Audio
from tqdm.auto import tqdm
import librosa
import numpy as np

def load_common_voice_swahili():
    """Load Swahili subset from Common Voice dataset."""
    print("Loading Common Voice Swahili dataset...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "sw",  # Swahili language code
        use_auth_token=True  # You need to be logged in to Hugging Face
    )
    return dataset

def process_audio(audio_path: str, target_sr: int = 22050) -> Tuple[torch.Tensor, float]:
    """Load and preprocess audio file."""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr

def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """Extract mel spectrogram from waveform."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=1.0,
        norm="slaney",
        mel_scale="slaney"
    )
    
    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log1p(mel_spec)
    
    return mel_spec

def prepare_dataset(
    dataset,
    output_dir: str,
    split: str = "train",
    max_samples: int = None
):
    """Prepare dataset for training."""
    # Create output directories
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Filter dataset
    dataset = dataset[split]
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Process samples
    metadata = []
    for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split} set")):
        try:
            # Create speaker directory
            speaker_id = f"Speaker_{idx:03d}"
            speaker_dir = os.path.join(split_dir, speaker_id)
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Process audio
            waveform = torch.tensor(item['audio']['array']).unsqueeze(0)
            sample_rate = item['audio']['sampling_rate']
            
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                waveform = resampler(waveform)
                sample_rate = 22050
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(waveform, sample_rate)
            
            # Save processed data
            torch.save(mel_spec, os.path.join(speaker_dir, 'mel.pt'))
            
            # Save text
            text = item['sentence']
            with open(os.path.join(speaker_dir, 'text.txt'), 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Collect metadata
            duration = waveform.shape[1] / sample_rate
            metadata.append({
                'speaker_id': speaker_id,
                'text': text,
                'duration': duration,
                'mel_frames': mel_spec.shape[1]
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(split_dir, 'metadata.csv'), index=False)
    
    print(f"\nProcessed {len(metadata)} samples")
    print(f"Total duration: {sum(m['duration'] for m in metadata) / 3600:.2f} hours")
    print(f"Average duration: {np.mean([m['duration'] for m in metadata]):.2f} seconds")

def main():
    """Main function."""
    # Load dataset
    dataset = load_common_voice_swahili()
    
    # Process splits
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split} split...")
        prepare_dataset(
            dataset,
            output_dir='processed_data',
            split=split,
            max_samples=None  # Set to a number to limit samples per split
        )

if __name__ == '__main__':
    main()
