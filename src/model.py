"""
FastSpeech 2 model implementation for Kenyan Swahili TTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class FFTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = residual + x

        # Feed forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x

        return x

class LengthRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.length_layer = nn.Linear(384, 1)  # d_model -> 1
        
    def forward(self, x, duration_target=None):
        """Expand encoder output according to predicted or target duration.
        
        Args:
            x (Tensor): Encoder output [B, T, C]
            duration_target (Tensor, optional): Target duration for training [B, T]
            
        Returns:
            Tensor: Expanded features [B, T', C]
            Tensor: Duration predictions [B, T]
        """
        duration_pred = self.length_layer(x).squeeze(-1)  # [B, T]
        
        if duration_target is not None:
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(duration_pred)), min=1)
        
        # Calculate expanded sequence lengths
        expand_length = torch.sum(duration_rounded, dim=1).long()  # [B]
        max_length = expand_length.max()
        
        # Initialize output tensor
        batch_size, _, channels = x.size()
        expanded = x.new_zeros((batch_size, max_length, channels))
        
        # Expand each batch item
        for i in range(batch_size):
            expanded_idx = 0
            for j, duration in enumerate(duration_rounded[i]):
                duration = int(duration.item())
                if duration > 0:  # Only expand if duration > 0
                    expanded[i, expanded_idx:expanded_idx + duration] = x[i, j].unsqueeze(0).expand(duration, -1)
                    expanded_idx += duration
        
        return expanded, duration_pred

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_head: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 n_head: int,
                 d_ff: int,
                 n_mels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        mel_output = self.mel_linear(x)
        return mel_output

class FastSpeech2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 384,
                 n_enc_layers: int = 4,
                 n_dec_layers: int = 4,
                 n_heads: int = 2,
                 d_ff: int = 1536,
                 n_mels: int = 80,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_enc_layers,
            n_head=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        self.length_regulator = LengthRegulator()
        
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_dec_layers,
            n_head=n_heads,
            d_ff=d_ff,
            n_mels=n_mels,
            dropout=dropout
        )

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                duration_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create padding mask
        src_pad_mask = (src != 0).unsqueeze(-2)  # [batch, 1, time]
        
        # Encode
        encoder_output = self.encoder(src)
        
        # Length regulation
        length_regulated, duration_pred = self.length_regulator(
            encoder_output,
            duration_target
        )
        
        # Decode
        mel_output = self.decoder(length_regulated)
        
        return mel_output, duration_pred

class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self,
                mel_output: torch.Tensor,
                duration_predicted: torch.Tensor,
                mel_target: torch.Tensor,
                duration_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Mel loss
        mel_loss = self.mse_loss(mel_output, mel_target)
        
        # Duration loss
        duration_loss = self.mae_loss(duration_predicted, torch.log1p(duration_target.float()))
        
        # Total loss
        total_loss = mel_loss + duration_loss
        
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'duration_loss': duration_loss.item(),
            'total_loss': total_loss.item()
        }
