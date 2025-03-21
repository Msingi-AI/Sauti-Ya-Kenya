"""
FastSpeech 2 model implementation for Kenyan Swahili TTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\nPositionalEncoding input shape: {x.shape}")
        print(f"pe shape: {self.pe.shape}")
        
        seq_len = x.size(1)
        if seq_len > self.max_len:
            print(f"Warning: Input sequence length {seq_len} exceeds maximum length {self.max_len}. Truncating.")
            x = x[:, :self.max_len]
            
        output = x + self.pe[:, :x.size(1)]
        print(f"PositionalEncoding output shape: {output.shape}")
        return output

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
        print(f"\nFFTBlock input shape: {x.shape}")
        
        # Self attention
        residual = x
        x = self.norm1(x)
        x = x.transpose(0, 1)  # [B, T, D] -> [T, B, D] for attention
        x, _ = self.self_attn(x, x, x)
        x = x.transpose(0, 1)  # [T, B, D] -> [B, T, D]
        x = self.dropout(x)
        x = residual + x

        # Feed forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x
        
        print(f"FFTBlock output shape: {x.shape}")
        return x

class LengthRegulator(nn.Module):
    def __init__(self, max_len: int = 10000):
        super().__init__()
        self.length_layer = nn.Linear(384, 1)  # d_model -> 1
        self.max_len = max_len
        
    def forward(self, x: torch.Tensor, duration_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        print(f"\nLengthRegulator input shapes:")
        print(f"x: {x.shape}")
        if duration_target is not None:
            print(f"duration_target: {duration_target.shape}")
        
        # Predict duration if not provided
        log_duration_pred = self.length_layer(x)
        duration_pred = torch.exp(log_duration_pred) - 1
        
        if duration_target is None:
            duration_rounded = torch.round(duration_pred)
        else:
            duration_rounded = duration_target
            
        # Calculate total expanded length
        expanded_len = int(duration_rounded.sum().item())
        if expanded_len > self.max_len:
            # Scale down durations to fit max_len
            scale_factor = self.max_len / expanded_len
            duration_rounded = torch.floor(duration_rounded * scale_factor)
            print(f"Warning: Expanded length {expanded_len} exceeds maximum length {self.max_len}. Scaling durations.")
            expanded_len = int(duration_rounded.sum().item())
            
        # Expand according to duration
        expanded = torch.zeros(x.size(0), expanded_len, x.size(2)).to(x.device)
        current_pos = 0
        for i in range(x.size(1)):
            expanded_size = int(duration_rounded[:, i].sum().item())
            if expanded_size > 0:
                expanded[:, current_pos:current_pos + expanded_size] = x[:, i:i+1].expand(-1, expanded_size, -1)
                current_pos += expanded_size
        
        print(f"\nLengthRegulator output shapes:")
        print(f"expanded: {expanded.shape}")
        print(f"duration_pred: {duration_pred.squeeze(-1).shape}")
                
        return expanded, duration_pred.squeeze(-1)

class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_head: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 max_len: int = 10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\nEncoder input shape: {x.shape}")
        
        # Embed tokens
        x = self.embedding(x)
        print(f"After embedding shape: {x.shape}")
        
        # Add positional encoding
        x = self.pos_encoder(x)
        print(f"After positional encoding shape: {x.shape}")
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        print(f"Encoder output shape: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_layers: int,
                 n_head: int,
                 d_ff: int,
                 n_mels: int,
                 dropout: float = 0.1,
                 max_len: int = 10000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\nDecoder input shape: {x.shape}")
        
        x = self.pos_encoder(x)
        print(f"After positional encoding shape: {x.shape}")
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        print(f"After final norm shape: {x.shape}")
        
        mel_output = self.mel_linear(x)
        print(f"Final mel output shape: {mel_output.shape}")
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
                 dropout: float = 0.1,
                 max_len: int = 10000):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_enc_layers,
            n_head=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        self.length_regulator = LengthRegulator(max_len=max_len)
        
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_dec_layers,
            n_head=n_heads,
            d_ff=d_ff,
            n_mels=n_mels,
            dropout=dropout,
            max_len=max_len
        )

    def forward(self,
                src: torch.Tensor,
                duration_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        print("\nFastSpeech2 input shapes:")
        print(f"src: {src.shape}")
        if duration_target is not None:
            print(f"duration_target: {duration_target.shape}")
        
        # Encode input sequence
        encoder_output = self.encoder(src)
        print(f"\nEncoder output shape: {encoder_output.shape}")
        
        # Length regulation
        length_regulated, duration_pred = self.length_regulator(encoder_output, duration_target)
        print(f"\nAfter length regulation shapes:")
        print(f"length_regulated: {length_regulated.shape}")
        print(f"duration_pred: {duration_pred.shape}")
        
        # Decode to generate mel spectrogram
        mel_output = self.decoder(length_regulated)
        print(f"\nDecoder output shape: {mel_output.shape}")
        
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
