"""
Tests for model configuration
"""
import pytest
from src.config import ModelConfig

def test_model_config():
    """Test model configuration defaults"""
    config = ModelConfig()
    
    # Test encoder settings
    assert config.vocab_size == 1000
    assert config.encoder_layers == 4
    assert config.encoder_dim == 256
    assert config.encoder_heads == 2
    
    # Test duration predictor settings
    assert config.duration_predictor_layers == 2
    assert config.duration_predictor_channels == 256
    
    # Test decoder settings
    assert config.decoder_layers == 4
    assert config.decoder_dim == 256
    assert config.decoder_heads == 2
    
    # Test audio settings
    assert config.sample_rate == 22050
    assert config.n_mel_channels == 80
    assert config.mel_fmin == 0.0
    assert config.mel_fmax == 8000.0
    
    # Test training settings
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.warmup_steps == 4000
