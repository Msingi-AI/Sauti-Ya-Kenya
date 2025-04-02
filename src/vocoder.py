"""
HiFiGAN vocoder for converting mel-spectrograms to waveforms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import List, Optional, Tuple
from einops import rearrange
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, 
                         padding=((kernel_size-1)*d)//2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                         padding=(kernel_size-1)//2)
            ) for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x

class MRF(nn.Module):
    """Multi-Receptive Field Fusion"""
    def __init__(self, channels: int, kernel_sizes: List[int] = [3, 7, 11]):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k) for k in kernel_sizes
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum([block(x) for block in self.resblocks]) / len(self.resblocks)

class Generator(nn.Module):
    def __init__(self, 
                 initial_channel: int = 512,
                 upsample_rates: List[int] = [8, 8, 2, 2],
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                 resblock_kernel_sizes: List[int] = [3, 7, 11],
                 resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        
        self.hop_length = np.prod(upsample_rates)  # Total upsampling factor
        
        # Initial conv to convert mel spectrogram to hidden features
        self.conv_pre = nn.Conv1d(80, initial_channel, 3, padding=1)
        
        # Upsample to audio sample rate
        self.ups = nn.ModuleList()
        curr_channel = initial_channel
        
        for i, (u_rate, u_k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate padding to maintain length after transposed conv
            padding = (u_k - u_rate) // 2
            
            self.ups.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        curr_channel, curr_channel // 2,
                        u_k, stride=u_rate,
                        padding=padding
                    )
                )
            )
            curr_channel //= 2
        
        # Apply MRF blocks for better audio quality
        self.mrf_blocks = nn.ModuleList([
            MRF(curr_channel, [3, 5, 7])  # Smaller kernel sizes
            for _ in range(3)  # Use 3 MRF blocks
        ])
        
        # Final conv to generate waveform
        self.conv_post = nn.Conv1d(curr_channel, 1, 3, padding=1)
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform
        Args:
            mel: (batch_size, n_mel_channels, time) or (n_mel_channels, time)
        Returns:
            waveform: (batch_size, 1, time * hop_length)
        """
        # Ensure mel is in the right format
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add batch dimension
        elif mel.dim() == 3 and mel.size(1) != 80:
            mel = mel.transpose(1, 2)  # Swap channels and time
            
        # Print shapes for debugging
        print(f"\nVocoder input shape: {mel.shape}")
        
        # Ensure minimum input length
        min_length = 8  # Minimum length needed for upsampling
        if mel.size(-1) < min_length:
            pad_length = min_length - mel.size(-1)
            mel = F.pad(mel, (0, pad_length), mode='replicate')
            print(f"Padded mel shape: {mel.shape}")
        
        # Generate waveform
        x = self.conv_pre(mel)
        print(f"After conv_pre shape: {x.shape}")
        
        # Normalize activation range
        x = F.instance_norm(x)
        
        # Upsample in stages
        for i, up in enumerate(self.ups):
            x = up(x)
            # Normalize after each upsampling
            x = F.instance_norm(x)
            print(f"After upsample {i+1} shape: {x.shape}")
        
        # Apply MRF blocks
        for i, mrf in enumerate(self.mrf_blocks):
            x = x + 0.1 * mrf(x)  # Scaled residual connection
            print(f"After MRF {i+1} shape: {x.shape}")
        
        # Final processing
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # Scale output to full range [-1, 1]
        max_val = x.abs().max()
        if max_val > 0:
            x = x / max_val
        
        print(f"Final waveform shape: {x.shape}")
        return x

class MelSpectrogram(nn.Module):
    """Mel-spectrogram transformation with customizable parameters"""
    def __init__(self,
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 win_length: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=mel_fmin,
            f_max=mel_fmax,
            center=True,
            power=1.0,  # Using amplitude spectrogram
            norm="slaney",
            mel_scale="slaney"
        )
        self.hop_length = hop_length
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel-spectrogram
        Args:
            audio: (batch_size, 1, time)
        Returns:
            mel: (batch_size, n_mels, time)
        """
        mel = self.mel_transform(audio.squeeze(1))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel

class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        channels = [32, 128, 512, 1024]
        kernel_size = 5
        stride = 3
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels[0], (kernel_size, 1), (stride, 1), padding=(2, 0)),
            *[nn.Conv2d(channels[i], channels[i + 1], (kernel_size, 1), (stride, 1), padding=(2, 0))
              for i in range(len(channels) - 1)]
        ])
        
        self.conv_post = nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, time = x.shape
        
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            time = time + n_pad
            
        x = rearrange(x, 'b c (t p) -> b 1 t p', p=self.period)
        x = rearrange(x, 'b c t p -> b 1 t p')
        
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = list(zip(*[d(x) for d in self.discriminators]))
        return ret[0], ret[1]  # [scores], [fmaps]

class HiFiGAN(nn.Module):
    """HiFiGAN vocoder for converting mel-spectrograms to waveforms"""
    def __init__(self, config=None):
        super().__init__()
        self.generator = Generator()
        self.mpd = MultiPeriodDiscriminator()
        self.mel_transform = MelSpectrogram(
            sample_rate=config.sample_rate if config else 22050,
            n_mels=config.n_mel_channels if config else 80,
            mel_fmin=config.mel_fmin if config else 0.0,
            mel_fmax=config.mel_fmax if config else 8000.0
        )
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform
        Args:
            mel: (batch_size, n_mel_channels, time)
        Returns:
            waveform: (batch_size, 1, time * hop_length)
        """
        return self.generator(mel)
    
    def train_step(self, mel: torch.Tensor, audio: torch.Tensor) -> dict:
        """
        Training step
        Args:
            mel: (batch_size, n_mel_channels, time)
            audio: (batch_size, 1, time)
        Returns:
            losses: dict of loss values
        """
        # Generator forward
        fake_audio = self.generator(mel)
        
        # Discriminator
        score_fake, fmap_fake = self.mpd(fake_audio.detach())
        score_real, fmap_real = self.mpd(audio)
        
        # Loss computation
        loss_disc = 0
        for sf, sr in zip(score_fake, score_real):
            loss_disc += torch.mean((1 - sr) ** 2) + torch.mean(sf ** 2)
            
        # Generator loss
        score_fake, fmap_fake = self.mpd(fake_audio)
        loss_gen = 0
        for sf in score_fake:
            loss_gen += torch.mean((1 - sf) ** 2)
            
        # Feature matching loss
        loss_fm = 0
        for fmap_f, fmap_r in zip(fmap_fake, fmap_real):
            for f_f, f_r in zip(fmap_f, fmap_r):
                loss_fm += torch.mean(torch.abs(f_f - f_r))
                
        # Mel-spectrogram loss
        loss_mel = F.l1_loss(mel, self.mel_transform(fake_audio))
        
        return {
            'loss_discriminator': loss_disc,
            'loss_generator': loss_gen,
            'loss_feature_matching': loss_fm,
            'loss_mel': loss_mel
        }
    
    def convert_mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to audio waveform
        Args:
            mel: (batch_size, n_mel_channels, time)
        Returns:
            audio: (batch_size, 1, time * hop_length)
        """
        self.eval()
        with torch.no_grad():
            audio = self.forward(mel)
        return audio

def load_hifigan(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load Universal HiFi-GAN vocoder"""
    model = HiFiGAN()
    
    # Initialize with default parameters
    model.eval()
    model.to(device)
    
    print("Using default HiFi-GAN vocoder")
    return model
