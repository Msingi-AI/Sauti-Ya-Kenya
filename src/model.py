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
    """Duration predictor for expanding encoder outputs"""
    def __init__(self, d_model: int):
        super().__init__()
        self.length_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensure positive durations
        )
        
    def forward(self, x: torch.Tensor, duration_target: Optional[torch.Tensor] = None):
        """
        Expand input sequence according to predicted or target durations
        Args:
            x: Input tensor (B, T, d_model)
            duration_target: Target durations (B, T) or None for inference
        Returns:
            expanded: Expanded tensor (B, T', d_model)
            duration_pred: Predicted durations (B, T)
        """
        # Print input shapes
        print("\nLengthRegulator input shapes:")
        print(f"x: {x.shape}")
        print(f"duration_target: {duration_target}")
        
        # Predict durations
        duration_pred = self.length_layer(x).squeeze(-1)  # [B, T]
        
        # During inference, ensure reasonable durations
        if duration_target is None:
            # Scale durations based on position
            # First token (usually BOS) - shorter
            # Middle tokens (content) - longer
            # Last token (usually EOS) - medium
            batch_size, seq_len = duration_pred.shape
            
            # Base duration for each position
            base_durations = torch.ones_like(duration_pred) * 20  # Default duration
            
            if seq_len > 2:  # If we have BOS, content, and EOS
                # BOS token - shorter duration
                base_durations[:, 0] = 10
                
                # Content tokens - longer duration
                base_durations[:, 1:-1] = 30
                
                # EOS token - medium duration
                base_durations[:, -1] = 15
            
            # Apply base durations and add some variance
            duration_pred = base_durations * (0.8 + 0.4 * torch.rand_like(duration_pred))
            duration_pred = torch.round(duration_pred)
        
        print(f"Predicted durations: {duration_pred[0].tolist()}")
        
        # Expand according to durations
        batch_size, max_length, channels = x.shape
        total_length = int(duration_pred.sum().item())
        expanded = torch.zeros(batch_size, total_length, channels).to(x.device)
        
        # Fill expanded tensor with interpolation
        cur_pos = 0
        for i in range(max_length):
            dur = int(duration_pred[0, i].item())
            if dur > 0:
                if i < max_length - 1:
                    # Interpolate between current and next frame
                    next_frame = x[:, min(i + 1, max_length - 1)]
                    for j in range(dur):
                        alpha = j / dur
                        expanded[:, cur_pos + j] = (1 - alpha) * x[:, i] + alpha * next_frame
                else:
                    # For last frame, just repeat
                    expanded[:, cur_pos:cur_pos + dur] = x[:, i:i + 1]
                cur_pos += dur
        
        # Print output shapes
        print("\nLengthRegulator output shapes:")
        print(f"expanded: {expanded.shape}")
        print(f"duration_pred: {duration_pred.shape}")
        
        return expanded, duration_pred

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
    """FastSpeech2 TTS model"""
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 384,
                 n_enc_layers: int = 4,
                 n_dec_layers: int = 4,
                 n_heads: int = 2,
                 d_ff: int = 1536,
                 dropout: float = 0.1,
                 n_mels: int = 80,
                 max_len: int = 10000):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Encoder
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # Length regulator
        self.length_regulator = LengthRegulator(d_model)
        
        # Decoder
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_dec_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        # Mel-spectrogram projection
        self.mel_linear = nn.Linear(d_model, n_mels)
        
    def forward(self, src: torch.Tensor, 
                duration_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        Args:
            src: Source tokens (B, T)
            duration_target: Duration targets for training (B, T) or None for inference
        Returns:
            mel_output: Mel-spectrogram (B, T', n_mels)
        """
        # Print input shapes
        print("\nFastSpeech2 input shapes:")
        print(f"src: {src.shape}")
        
        # Embedding
        x = self.embedding(src)  # [B, T, d_model]
        print("After embedding shape:", x.shape)
        
        # Encode
        print("\nEncoder input shape:", x.shape)
        x = self.encoder(x)
        print("Encoder output shape:", x.shape)
        
        # Length regulation
        print("\nAfter length regulation shapes:")
        x, duration_pred = self.length_regulator(x, duration_target)
        print(f"length_regulated: {x.shape}")
        print(f"duration_pred: {duration_pred.shape}")
        
        # Decode
        print("\nDecoder input shape:", x.shape)
        x = self.decoder(x)
        print("After final norm shape:", x.shape)
        
        # Project to mel-spectrogram
        mel_output = self.mel_linear(x)  # [B, T', 80]
        mel_output = mel_output.transpose(1, 2)  # [B, 80, T']
        print("Final mel output shape:", mel_output.shape)
        
        return mel_output

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
