"""
HiFiGAN vocoder for high-quality speech synthesis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from einops import rearrange

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
        
        self.conv_pre = nn.Conv1d(80, initial_channel, 7, padding=3)  # 80 = mel channels
        
        channels = initial_channel
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.ConvTranspose1d(channels // (1 if i == 0 else 2),
                                 channels // 2,
                                 k, stride=u, padding=(k-u)//2)
            ))
            channels = channels // 2
            
        self.mrf_blocks = nn.ModuleList([
            MRF(channels, resblock_kernel_sizes) 
            for _ in range(len(resblock_dilations))
        ])
        
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        
        for up in self.ups:
            x = up(x)
            
        for mrf in self.mrf_blocks:
            x = mrf(x)
            
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

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
            
        x = rearrange(x, 'b c (t p) -> b c t p', p=self.period)
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
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.mpd = MultiPeriodDiscriminator()
        
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
    
    @staticmethod
    def mel_transform(audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel-spectrogram using torchaudio"""
        # This will be implemented using torchaudio's MelSpectrogram
        pass
