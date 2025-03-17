"""
Data augmentation for TTS training
"""
import torch
import torch.nn.functional as F
import torchaudio
import random
import numpy as np
from typing import Tuple, Optional

class AudioAugmenter:
    """Audio augmentation for TTS training"""
    def __init__(self,
                 sample_rate: int = 22050,
                 noise_factor: float = 0.005,
                 speed_range: Tuple[float, float] = (0.9, 1.1),
                 pitch_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Initialize audio augmenter
        Args:
            sample_rate: Audio sample rate
            noise_factor: Maximum noise amplitude
            speed_range: Range for speed perturbation
            pitch_range: Range for pitch shifting
        """
        self.sample_rate = sample_rate
        self.noise_factor = noise_factor
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        
    def add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio"""
        noise = torch.randn_like(audio) * self.noise_factor
        return audio + noise
    
    def change_speed(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random speed perturbation"""
        speed_factor = random.uniform(*self.speed_range)
        
        # Resample audio
        old_length = audio.size(-1)
        new_length = int(old_length / speed_factor)
        
        return F.interpolate(
            audio.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    def shift_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shifting"""
        pitch_factor = random.uniform(*self.pitch_range)
        
        effects = [
            ["pitch", str(int((pitch_factor - 1.0) * 100))],
            ["rate", str(self.sample_rate)]
        ]
        
        audio_np = audio.numpy()
        pitched_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            torch.from_numpy(audio_np),
            self.sample_rate,
            effects
        )
        
        return pitched_audio

class SpecAugment:
    """SpecAugment for mel-spectrogram augmentation"""
    def __init__(self,
                 freq_mask_param: int = 30,
                 time_mask_param: int = 40,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2):
        """
        Initialize SpecAugment
        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def apply_freq_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking"""
        B, C, F, T = mel.shape
        
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, F - f)
            
            mel[:, :, f0:f0 + f, :] = 0
            
        return mel
    
    def apply_time_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply time masking"""
        B, C, F, T = mel.shape
        
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, T - t)
            
            mel[:, :, :, t0:t0 + t] = 0
            
        return mel
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment"""
        mel = self.apply_freq_mask(mel)
        mel = self.apply_time_mask(mel)
        return mel

class TextAugmenter:
    """Text augmentation for TTS training"""
    def __init__(self,
                 swap_prob: float = 0.1,
                 delete_prob: float = 0.1,
                 substitute_prob: float = 0.1):
        """
        Initialize text augmenter
        Args:
            swap_prob: Probability of swapping adjacent tokens
            delete_prob: Probability of deleting a token
            substitute_prob: Probability of substituting a token
        """
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.substitute_prob = substitute_prob
        
    def swap_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Randomly swap adjacent tokens"""
        if len(tokens) < 2:
            return tokens
            
        result = tokens.clone()
        for i in range(len(tokens) - 1):
            if random.random() < self.swap_prob:
                result[i], result[i + 1] = result[i + 1], result[i]
                
        return result
    
    def delete_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Randomly delete tokens"""
        mask = torch.rand(len(tokens)) >= self.delete_prob
        if not mask.any():  # Ensure at least one token remains
            mask[0] = True
        return tokens[mask]
    
    def substitute_tokens(self,
                        tokens: torch.Tensor,
                        vocab_size: int) -> torch.Tensor:
        """Randomly substitute tokens"""
        result = tokens.clone()
        for i in range(len(tokens)):
            if random.random() < self.substitute_prob:
                result[i] = random.randint(0, vocab_size - 1)
        return result
    
    def __call__(self,
                 tokens: torch.Tensor,
                 vocab_size: int) -> torch.Tensor:
        """Apply text augmentation"""
        tokens = self.swap_tokens(tokens)
        tokens = self.delete_tokens(tokens)
        tokens = self.substitute_tokens(tokens, vocab_size)
        return tokens

class AugmentationPipeline:
    """Complete augmentation pipeline for TTS training"""
    def __init__(self,
                 audio_augmenter: Optional[AudioAugmenter] = None,
                 spec_augment: Optional[SpecAugment] = None,
                 text_augmenter: Optional[TextAugmenter] = None):
        """
        Initialize augmentation pipeline
        Args:
            audio_augmenter: Audio augmentation
            spec_augment: Spectrogram augmentation
            text_augmenter: Text augmentation
        """
        self.audio_augmenter = audio_augmenter or AudioAugmenter()
        self.spec_augment = spec_augment or SpecAugment()
        self.text_augmenter = text_augmenter or TextAugmenter()
        
    def augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation"""
        if random.random() < 0.5:
            audio = self.audio_augmenter.add_noise(audio)
        if random.random() < 0.5:
            audio = self.audio_augmenter.change_speed(audio)
        if random.random() < 0.5:
            audio = self.audio_augmenter.shift_pitch(audio)
        return audio
    
    def augment_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply mel-spectrogram augmentation"""
        if random.random() < 0.5:
            mel = self.spec_augment(mel)
        return mel
    
    def augment_text(self,
                    tokens: torch.Tensor,
                    vocab_size: int) -> torch.Tensor:
        """Apply text augmentation"""
        if random.random() < 0.5:
            tokens = self.text_augmenter(tokens, vocab_size)
        return tokens
    
    def __call__(self,
                 audio: torch.Tensor,
                 mel: torch.Tensor,
                 tokens: torch.Tensor,
                 vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply complete augmentation pipeline"""
        audio = self.augment_audio(audio)
        mel = self.augment_mel(mel)
        tokens = self.augment_text(tokens, vocab_size)
        return audio, mel, tokens
