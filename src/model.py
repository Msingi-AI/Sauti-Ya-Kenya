"""
Core TTS model implementation for Sauti Ya Kenya using FastSpeech2 architecture
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class FFTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_k: int, d_v: int, conv_kernel_size: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.conv1 = nn.Conv1d(d_model, d_model * 4, 1)
        self.conv2 = nn.Conv1d(d_model * 4, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=(conv_kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, conv_kernel_size, padding=(conv_kernel_size-1)//2),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed Forward
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        
        return x

class DurationPredictor(nn.Module):
    def __init__(self, channels: int, kernel_size: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2),
                nn.ReLU(),
                nn.LayerNorm(channels),
                nn.Dropout(0.1)
            ) for _ in range(layers)
        ])
        self.proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x.squeeze(-1)

class KenyanSwahiliTTS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder
        self.encoder_embedding = nn.Embedding(config.vocab_size, config.encoder_dim)
        self.encoder_prenet = nn.Linear(config.encoder_dim, config.encoder_dim)
        self.encoder_pos = PositionalEncoding(config.encoder_dim)
        self.encoder_layers = nn.ModuleList([
            FFTBlock(config.encoder_dim, config.encoder_heads, 
                    config.encoder_dim // config.encoder_heads,
                    config.encoder_dim // config.encoder_heads,
                    config.encoder_conv_kernel)
            for _ in range(config.encoder_layers)
        ])
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            config.duration_predictor_channels,
            config.duration_predictor_kernel,
            config.duration_predictor_layers
        )
        
        # Mel-spectrogram decoder
        self.decoder_prenet = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.decoder_pos = PositionalEncoding(config.decoder_dim)
        self.decoder_layers = nn.ModuleList([
            FFTBlock(config.decoder_dim, config.decoder_heads,
                    config.decoder_dim // config.decoder_heads,
                    config.decoder_dim // config.decoder_heads,
                    config.decoder_conv_kernel)
            for _ in range(config.decoder_layers)
        ])
        
        # Final projection to mel-spectrogram
        self.mel_proj = nn.Linear(config.decoder_dim, config.n_mel_channels)

    def forward(self, text_encodings: torch.Tensor, 
                durations: Optional[torch.Tensor] = None,
                mel_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TTS model
        Args:
            text_encodings: (batch_size, text_len)
            durations: (batch_size, text_len) for training
            mel_target: (batch_size, mel_len, n_mel_channels) for training
        Returns:
            mel_output: (batch_size, mel_len, n_mel_channels)
            duration_pred: (batch_size, text_len)
        """
        # Encode text
        x = self.encoder_embedding(text_encodings)
        x = self.encoder_prenet(x)
        x = self.encoder_pos(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Predict durations
        duration_pred = self.duration_predictor(x)
        if self.training:
            assert durations is not None, "Durations required for training"
            duration_rounded = durations
        else:
            duration_rounded = torch.round(torch.exp(duration_pred)).long()
        
        # Expand encodings according to duration
        expanded = []
        for batch_idx in range(x.size(0)):
            expanded.append(self._expand_encodings(
                x[batch_idx], duration_rounded[batch_idx]))
        x = torch.stack(expanded)
        
        # Decode to mel-spectrogram
        x = self.decoder_prenet(x)
        x = self.decoder_pos(x)
        
        for layer in self.decoder_layers:
            x = layer(x)
        
        mel_output = self.mel_proj(x)
        
        return mel_output, duration_pred

    def _expand_encodings(self, encoded: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """Expand encoded tokens according to predicted/target durations"""
        out = []
        for i, duration in enumerate(durations):
            out.append(encoded[i].unsqueeze(0).repeat(duration, 1))
        return torch.cat(out, dim=0)

    def generate(self, text_encodings: torch.Tensor) -> torch.Tensor:
        """
        Generate mel-spectrogram from text
        Args:
            text_encodings: (batch_size, text_len)
        Returns:
            mel_output: (batch_size, mel_len, n_mel_channels)
        """
        self.eval()
        with torch.no_grad():
            mel_output, _ = self.forward(text_encodings)
        return mel_output
