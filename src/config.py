"""
Model and training configuration
"""
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Text encoder
    vocab_size: int = 1000  # Will be updated based on tokenizer
    encoder_layers: int = 4
    encoder_dim: int = 256
    encoder_heads: int = 2
    encoder_conv_kernel: int = 9
    
    # Duration predictor
    duration_predictor_layers: int = 2
    duration_predictor_channels: int = 256
    duration_predictor_kernel: int = 3
    
    # Mel-spectrogram decoder
    decoder_layers: int = 4
    decoder_dim: int = 256
    decoder_heads: int = 2
    decoder_conv_kernel: int = 9
    
    # Audio settings
    sample_rate: int = 22050
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    warmup_steps: int = 4000
