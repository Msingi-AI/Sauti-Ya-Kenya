"""
HiFi-GAN vocoder implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class ResBlock(nn.Module):
    """Residual block for the generator"""
    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding='same'),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding='same')
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
        return sum(block(x) for block in self.resblocks) / len(self.resblocks)

class Generator(nn.Module):
    """HiFi-GAN Generator"""
    def __init__(self, 
                 in_channels: int = 80,
                 upsample_rates: List[int] = [8, 8, 2, 2],
                 upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
                 hidden_channels: int = 512,
                 resblock_kernel_sizes: List[int] = [3, 7, 11]):
        super().__init__()
        
        # Initial conv
        self.conv_pre = nn.Conv1d(in_channels, hidden_channels, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        curr_channels = hidden_channels
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.ConvTranspose1d(curr_channels, curr_channels // 2, k, stride=u, padding=(k-u)//2)
            ))
            curr_channels //= 2
            
        # MRF blocks
        self.mrfs = nn.ModuleList([
            MRF(curr_channels, resblock_kernel_sizes) for _ in range(4)
        ])
        
        # Output conv
        self.conv_post = nn.Conv1d(curr_channels, 1, 7, padding=3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv_pre(x)
        
        # Upsampling
        for up in self.ups:
            x = up(x)
            
        # MRF blocks
        for mrf in self.mrfs:
            x = mrf(x)
            
        # Output
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x

class PeriodDiscriminator(nn.Module):
    """Period-based discriminator"""
    def __init__(self, period: int):
        super().__init__()
        
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))
        ])
        
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.size(0)
        
        # Convert to 2D
        pad_size = (self.period - (x.size(-1) % self.period)) % self.period
        x = F.pad(x, (0, pad_size))
        x = x.view(batch_size, 1, -1, self.period)
        
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
    """Multi-period discriminator"""
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        feature_maps = []
        
        for disc in self.discriminators:
            out, fmap = disc(x)
            outputs.append(out)
            feature_maps.append(fmap)
            
        return outputs, feature_maps

class ScaleDiscriminator(nn.Module):
    """Scale discriminator"""
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2)
        ])
        
        self.conv_post = nn.Conv1d(1024, 1, 3, 1, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        
        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator"""
    def __init__(self):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(3)
        ])
        
        self.pooling = nn.AvgPool1d(4, 2, padding=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        feature_maps = []
        
        for disc in self.discriminators:
            out, fmap = disc(x)
            outputs.append(out)
            feature_maps.append(fmap)
            x = self.pooling(x)
            
        return outputs, feature_maps

class HiFiGAN(nn.Module):
    """HiFi-GAN vocoder"""
    def __init__(self):
        super().__init__()
        
        self.generator = Generator()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.generator(mel)
    
    def discriminate(self, audio: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # MPD
        mpd_out, mpd_fmap = self.mpd(audio)
        
        # MSD
        msd_out, msd_fmap = self.msd(audio)
        
        return mpd_out + msd_out, mpd_fmap + msd_fmap
