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
    def __init__(self, sample_rate, time_stretch_factors, pitch_shift_steps, volume_factors, speed_factors, noise_levels):
        self.sample_rate = sample_rate
        self.time_stretch_factors = time_stretch_factors
        self.pitch_shift_steps = pitch_shift_steps
        self.volume_factors = volume_factors
        self.speed_factors = speed_factors
        self.noise_levels = noise_levels
    
    def time_stretch(self, waveform, factor):
        if abs(factor - 1.0) < 1e-3:
            return waveform
            
        audio_np = waveform.numpy()[0]
        stretched = librosa.effects.time_stretch(audio_np, rate=factor)
        return torch.from_numpy(stretched).unsqueeze(0)
    
    def pitch_shift(self, waveform, steps):
        if steps == 0:
            return waveform
            
        audio_np = waveform.numpy()[0]
        shifted = librosa.effects.pitch_shift(audio_np, sr=self.sample_rate, n_steps=steps)
        return torch.from_numpy(shifted).unsqueeze(0)
    
    def change_volume(self, waveform, factor):
        return waveform * factor
    
    def change_speed(self, waveform, factor):
        if abs(factor - 1.0) < 1e-3:
            return waveform
            
        old_length = waveform.shape[1]
        new_length = int(old_length / factor)
        
        resampled = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=int(self.sample_rate * factor))(waveform)
        
        return resampled
    
    def add_noise(self, waveform, noise_level):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def augment_waveform(self, waveform, augmentation, params):
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

def augment_dataset(data_dir, output_dir, num_augmentations, augmenter):
    if augmenter is None:
        augmenter = AudioAugmenter(22050, [0.9, 1.1], [-2, -1, 1, 2], [0.8, 1.2], [0.9, 1.1], [0.001, 0.002, 0.003])
    
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    augmentation_types = [
        ("time_stretch", {"factor": augmenter.time_stretch_factors}),
        ("pitch_shift", {"steps": augmenter.pitch_shift_steps}),
        ("volume", {"factor": augmenter.volume_factors}),
        ("speed", {"factor": augmenter.speed_factors}),
        ("noise", {"level": augmenter.noise_levels})
    ]
    
    new_metadata = []
    
    for speaker_id in tqdm(metadata_df['speaker_id'].unique()):
        speaker_data = metadata_df[metadata_df['speaker_id'] == speaker_id]
        
        speaker_dir = os.path.join(output_dir, speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        for _, row in speaker_data.iterrows():
            mel_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}_mel.pt")
            text_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}_text.txt")
            
            if not os.path.exists(mel_path) or not os.path.exists(text_path):
                continue
            
            torch.save(torch.load(mel_path), os.path.join(speaker_dir, f"{row['clip_id']}_mel.pt"))
            with open(text_path, 'r') as f:
                text = f.read()
            with open(os.path.join(speaker_dir, f"{row['clip_id']}_text.txt"), 'w') as f:
                f.write(text)
            
            new_metadata.append(row.to_dict())
            
            for aug_idx in range(num_augmentations):
                try:
                    aug_type, params_dict = random.choice(augmentation_types)
                    params = {k: random.choice(v) if isinstance(v, list) else v for k, v in params_dict.items()}
                    
                    audio_path = os.path.join(data_dir, row['speaker_id'], f"{row['clip_id']}.wav")
                    if not os.path.exists(audio_path):
                        continue
                    
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != augmenter.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, augmenter.sample_rate)
                        waveform = resampler(waveform)
                    
                    aug_waveform, aug_name = augmenter.augment_waveform(waveform, aug_type, params)
                    
                    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=augmenter.sample_rate, n_fft=1024, hop_length=256, n_mels=80, center=True, power=1.0, norm="slaney", mel_scale="slaney")
                    
                    mel_spec = mel_transform(aug_waveform)
                    mel_spec = torch.log1p(mel_spec)
                    
                    aug_id = f"{row['clip_id']}_{aug_name}"
                    torch.save(mel_spec, os.path.join(speaker_dir, f"{aug_id}_mel.pt"))
                    with open(os.path.join(speaker_dir, f"{aug_id}_text.txt"), 'w') as f:
                        f.write(text)
                    
                    aug_metadata = row.to_dict()
                    aug_metadata.update({'clip_id': aug_id, 'augmentation': aug_name, 'original_clip_id': row['clip_id']})
                    new_metadata.append(aug_metadata)
                    
                except Exception as e:
                    continue
    
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Augment TTS dataset")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_augmentations", type=int, default=3)
    
    args = parser.parse_args()
    
    augment_dataset(args.data_dir, args.output_dir, args.num_augmentations)

if __name__ == '__main__':
    main()
