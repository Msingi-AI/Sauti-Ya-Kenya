"""
Data augmentation for TTS training data.
Implements various augmentation techniques to expand the dataset:
1. Time stretching
2. Pitch shifting
3. Adding background noise
4. Speed perturbation
5. Volume perturbation
"""
import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import random
import pandas as pd
from pathlib import Path
import librosa

class AudioAugmenter:
    def __init__(
        self,
        sample_rate: int = 22050,
        time_stretch_factors: List[float] = [0.9, 1.1],
        pitch_shift_steps: List[int] = [-2, -1, 1, 2],
        volume_factors: List[float] = [0.8, 1.2],
        speed_factors: List[float] = [0.9, 1.1],
        noise_levels: List[float] = [0.001, 0.002, 0.003]
    ):
        """Initialize augmenter with parameters.
        
        Args:
            sample_rate: Target sample rate
            time_stretch_factors: List of time stretching factors
            pitch_shift_steps: List of pitch shift steps (in semitones)
            volume_factors: List of volume multiplication factors
            speed_factors: List of speed factors
            noise_levels: List of noise standard deviations
        """
        self.sample_rate = sample_rate
        self.time_stretch_factors = time_stretch_factors
        self.pitch_shift_steps = pitch_shift_steps
        self.volume_factors = volume_factors
        self.speed_factors = speed_factors
        self.noise_levels = noise_levels
    
    def time_stretch(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply time stretching."""
        if abs(factor - 1.0) < 1e-3:
            return waveform
            
        audio_np = waveform.numpy()[0]
        stretched = librosa.effects.time_stretch(audio_np, rate=factor)
        return torch.from_numpy(stretched).unsqueeze(0)
    
    def pitch_shift(self, waveform: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply pitch shifting."""
        if steps == 0:
            return waveform
            
        audio_np = waveform.numpy()[0]
        shifted = librosa.effects.pitch_shift(
            audio_np, 
            sr=self.sample_rate,
            n_steps=steps
        )
        return torch.from_numpy(shifted).unsqueeze(0)
    
    def change_volume(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply volume change."""
        return waveform * factor
    
    def change_speed(self, waveform: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply speed change using resampling."""
        if abs(factor - 1.0) < 1e-3:
            return waveform
            
        old_length = waveform.shape[1]
        new_length = int(old_length / factor)
        
        # Resample to new length
        resampled = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * factor)
        )(waveform)
        
        return resampled
    
    def add_noise(self, waveform: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def augment_waveform(
        self,
        waveform: torch.Tensor,
        augmentation: str,
        params: Dict
    ) -> Tuple[torch.Tensor, str]:
        """Apply a single augmentation."""
        if augmentation == "time_stretch":
            factor = params["factor"]
            return self.time_stretch(waveform, factor), f"time_stretch_{factor:.2f}"
            
        elif augmentation == "pitch_shift":
            steps = params["steps"]
            return self.pitch_shift(waveform, steps), f"pitch_shift_{steps}"
            
        elif augmentation == "volume":
            factor = params["factor"]
            return self.change_volume(waveform, factor), f"volume_{factor:.2f}"
            
        elif augmentation == "speed":
            factor = params["factor"]
            return self.change_speed(waveform, factor), f"speed_{factor:.2f}"
            
        elif augmentation == "noise":
            level = params["level"]
            return self.add_noise(waveform, level), f"noise_{level:.3f}"
            
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")

def augment_dataset(
    data_dir: str,
    output_dir: str,
    num_augmentations: int = 3,
    augmenter: Optional[AudioAugmenter] = None
):
    """Augment all audio files in the dataset.
    
    Args:
        data_dir: Directory containing processed data
        output_dir: Directory to save augmented data
        num_augmentations: Number of augmented versions to create per clip
        augmenter: AudioAugmenter instance, or None to use default
    """
    if augmenter is None:
        augmenter = AudioAugmenter()
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    print(f"\nFound {len(metadata_df)} original clips")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of possible augmentations
    augmentation_types = [
        ("time_stretch", {"factor": augmenter.time_stretch_factors}),
        ("pitch_shift", {"steps": augmenter.pitch_shift_steps}),
        ("volume", {"factor": augmenter.volume_factors}),
        ("speed", {"factor": augmenter.speed_factors}),
        ("noise", {"level": augmenter.noise_levels})
    ]
    
    new_metadata = []
    
    # Process each speaker
    for speaker_id in tqdm(metadata_df['speaker_id'].unique(), desc="Processing speakers"):
        speaker_data = metadata_df[metadata_df['speaker_id'] == speaker_id]
        
        # Create speaker directory
        speaker_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Process each clip
        for _, row in speaker_data.iterrows():
            # Load original mel spectrogram
            mel_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}_mel.pt")
            text_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}_text.txt")
            
            if not os.path.exists(mel_path) or not os.path.exists(text_path):
                print(f"\nWarning: Missing files for {mel_path}")
                continue
            
            # Copy original files
            os.makedirs(speaker_dir, exist_ok=True)
            torch.save(torch.load(mel_path), os.path.join(speaker_dir, f"{row['clip_id']}_mel.pt"))
            with open(text_path, 'r') as f:
                text = f.read()
            with open(os.path.join(speaker_dir, f"{row['clip_id']}_text.txt"), 'w') as f:
                f.write(text)
            
            # Add original to metadata
            new_metadata.append(row.to_dict())
            
            # Create augmented versions
            for aug_idx in range(num_augmentations):
                try:
                    # Randomly select augmentations
                    aug_type, params_dict = random.choice(augmentation_types)
                    params = {
                        k: random.choice(v) if isinstance(v, list) else v
                        for k, v in params_dict.items()
                    }
                    
                    # Load and augment audio
                    audio_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}.wav")
                    if not os.path.exists(audio_path):
                        # Try to load mel and convert back to audio (approximate)
                        mel = torch.load(mel_path)
                        # TODO: Implement mel to audio conversion
                        continue
                    
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != augmenter.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, augmenter.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Apply augmentation
                    aug_waveform, aug_name = augmenter.augment_waveform(waveform, aug_type, params)
                    
                    # Extract mel spectrogram
                    mel_transform = torchaudio.transforms.MelSpectrogram(
                        sample_rate=augmenter.sample_rate,
                        n_fft=1024,
                        hop_length=256,
                        n_mels=80,
                        center=True,
                        power=1.0,
                        norm="slaney",
                        mel_scale="slaney"
                    )
                    
                    mel_spec = mel_transform(aug_waveform)
                    mel_spec = torch.log1p(mel_spec)
                    
                    # Save augmented data
                    aug_id = f"{row['clip_id']}_{aug_name}"
                    torch.save(mel_spec, os.path.join(speaker_dir, f"{aug_id}_mel.pt"))
                    with open(os.path.join(speaker_dir, f"{aug_id}_text.txt"), 'w') as f:
                        f.write(text)
                    
                    # Add to metadata
                    aug_metadata = row.to_dict()
                    aug_metadata.update({
                        'clip_id': aug_id,
                        'augmentation': aug_name,
                        'original_clip_id': row['clip_id']
                    })
                    new_metadata.append(aug_metadata)
                    
                except Exception as e:
                    print(f"\nError augmenting {row['clip_id']}: {str(e)}")
                    continue
    
    # Save new metadata
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"\nCreated {len(new_metadata_df) - len(metadata_df)} augmented clips")
    print(f"Total clips in augmented dataset: {len(new_metadata_df)}")

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Augment TTS dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save augmented data")
    parser.add_argument("--num_augmentations", type=int, default=3,
                      help="Number of augmented versions to create per clip")
    
    args = parser.parse_args()
    
    print("\nAugmenting dataset...")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentations per clip: {args.num_augmentations}")
    
    augment_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations
    )

if __name__ == '__main__':
    main()
