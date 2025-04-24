import torch
import torch.nn.functional as F
import torchaudio
import random
import numpy as np
from typing import Tuple, Optional

class AudioAugmenter:
    def __init__(self, sample_rate=22050, noise_factor=0.005, speed_range=(0.9, 1.1), pitch_range=(0.95, 1.05)):
        self.sample_rate = sample_rate
        self.noise_factor = noise_factor
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        
    def add_noise(self, audio):
        noise = torch.randn_like(audio) * self.noise_factor
        return audio + noise
    
    def change_speed(self, audio):
        speed_factor = random.uniform(*self.speed_range)
        
        old_length = audio.size(-1)
        new_length = int(old_length / speed_factor)
        
        return F.interpolate(
            audio.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    def shift_pitch(self, audio):
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
    def __init__(self, freq_mask_param=30, time_mask_param=40, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        
    def apply_freq_mask(self, mel):
        B, C, F, T = mel.shape
        
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, F - f)
            
            mel[:, :, f0:f0 + f, :] = 0
            
        return mel
    
    def apply_time_mask(self, mel):
        B, C, F, T = mel.shape
        
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, T - t)
            
            mel[:, :, :, t0:t0 + t] = 0
            
        return mel
    
    def __call__(self, mel):
        mel = self.apply_freq_mask(mel)
        mel = self.apply_time_mask(mel)
        return mel

class TextAugmenter:
    def __init__(self, swap_prob=0.1, delete_prob=0.1, substitute_prob=0.1):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.substitute_prob = substitute_prob
        
    def swap_tokens(self, tokens):
        if len(tokens) < 2:
            return tokens
            
        result = tokens.clone()
        for i in range(len(tokens) - 1):
            if random.random() < self.swap_prob:
                result[i], result[i + 1] = result[i + 1], result[i]
                
        return result
    
    def delete_tokens(self, tokens):
        mask = torch.rand(len(tokens)) >= self.delete_prob
        if not mask.any():  
            mask[0] = True
        return tokens[mask]
    
    def substitute_tokens(self, tokens, vocab_size):
        result = tokens.clone()
        for i in range(len(tokens)):
            if random.random() < self.substitute_prob:
                result[i] = random.randint(0, vocab_size - 1)
        return result
    
    def __call__(self, tokens, vocab_size):
        tokens = self.swap_tokens(tokens)
        tokens = self.delete_tokens(tokens)
        tokens = self.substitute_tokens(tokens, vocab_size)
        return tokens

class AugmentationPipeline:
    def __init__(self, audio_augmenter=None, spec_augment=None, text_augmenter=None):
        self.audio_augmenter = audio_augmenter or AudioAugmenter()
        self.spec_augment = spec_augment or SpecAugment()
        self.text_augmenter = text_augmenter or TextAugmenter()
        
    def augment_audio(self, audio):
        if random.random() < 0.5:
            audio = self.audio_augmenter.add_noise(audio)
        if random.random() < 0.5:
            audio = self.audio_augmenter.change_speed(audio)
        if random.random() < 0.5:
            audio = self.audio_augmenter.shift_pitch(audio)
        return audio
    
    def augment_mel(self, mel):
        if random.random() < 0.5:
            mel = self.spec_augment(mel)
        return mel
    
    def augment_text(self, tokens, vocab_size):
        if random.random() < 0.5:
            tokens = self.text_augmenter(tokens, vocab_size)
        return tokens
    
    def __call__(self, audio, mel, tokens, vocab_size):
        audio = self.augment_audio(audio)
        mel = self.augment_mel(mel)
        tokens = self.augment_text(tokens, vocab_size)
        return audio, mel, tokens
