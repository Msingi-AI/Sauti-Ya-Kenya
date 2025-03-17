"""
Tests for TTS model architecture
"""
import pytest
import torch
from src.config import ModelConfig
from src.model import KenyanSwahiliTTS

@pytest.fixture
def small_config():
    """Create a small model config for testing"""
    config = ModelConfig()
    # Reduce model size for faster testing
    config.encoder_layers = 2
    config.encoder_dim = 64
    config.encoder_heads = 2
    config.duration_predictor_layers = 1
    config.duration_predictor_channels = 64
    config.decoder_layers = 2
    config.decoder_dim = 64
    config.decoder_heads = 2
    return config

@pytest.fixture
def model(small_config):
    """Create a small model for testing"""
    return KenyanSwahiliTTS(small_config)

def test_encoder_output_shape(model):
    """Test encoder output dimensions"""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Get encoder output
    x = model.encoder_embedding(input_ids)
    x = model.encoder_prenet(x)
    x = model.encoder_pos(x)
    
    for layer in model.encoder_layers:
        x = layer(x)
    
    assert x.shape == (batch_size, seq_len, model.config.encoder_dim)

def test_duration_predictor(model):
    """Test duration predictor output"""
    batch_size = 2
    seq_len = 10
    encoder_output = torch.randn(batch_size, seq_len, model.config.encoder_dim)
    
    durations = model.duration_predictor(encoder_output)
    assert durations.shape == (batch_size, seq_len)

def test_forward_pass(model):
    """Test complete forward pass"""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    durations = torch.randint(1, 4, (batch_size, seq_len))
    
    mel_output, duration_pred = model(input_ids, durations)
    assert isinstance(mel_output, torch.Tensor)
    assert isinstance(duration_pred, torch.Tensor)
    assert duration_pred.shape == (batch_size, seq_len)

def test_inference_mode(model):
    """Test model behavior in inference mode"""
    model.eval()
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    with torch.no_grad():
        mel_output = model.generate(input_ids)
        assert isinstance(mel_output, torch.Tensor)
        assert mel_output.dim() == 3  # (batch, time, mels)
