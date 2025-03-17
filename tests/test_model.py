"""
Tests for TTS model architecture
"""
import pytest
import torch
import numpy as np
from src.model import KenyanSwahiliTTS
from src.config import ModelConfig

@pytest.fixture
def config():
    return ModelConfig()

@pytest.fixture
def model(config):
    return KenyanSwahiliTTS(config).eval()

def test_encoder_output_shape(model):
    """Test encoder output dimensions"""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    encoder_output = model.encoder(input_ids)
    assert encoder_output.shape == (batch_size, seq_len, model.config.hidden_size)

def test_duration_predictor(model):
    """Test duration predictor output"""
    batch_size = 2
    seq_len = 10
    hidden_size = model.config.hidden_size
    encoder_output = torch.randn(batch_size, seq_len, hidden_size)
    
    durations = model.duration_predictor(encoder_output)
    assert durations.shape == (batch_size, seq_len)
    assert torch.all(durations >= 0)  # Durations should be non-negative

def test_length_regulator(model):
    """Test length regulation"""
    batch_size = 2
    seq_len = 10
    hidden_size = model.config.hidden_size
    encoder_output = torch.randn(batch_size, seq_len, hidden_size)
    durations = torch.randint(1, 4, (batch_size, seq_len))
    
    regulated = model.length_regulator(encoder_output, durations)
    expected_length = durations.sum(dim=1).max().item()
    assert regulated.shape[1] == expected_length

def test_mel_decoder(model):
    """Test mel-spectrogram decoder"""
    batch_size = 2
    seq_len = 50
    hidden_size = model.config.hidden_size
    regulated = torch.randn(batch_size, seq_len, hidden_size)
    
    mel_output = model.mel_decoder(regulated)
    assert mel_output.shape == (batch_size, model.config.n_mel_channels, seq_len)

def test_forward_pass(model):
    """Test complete forward pass"""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    mel_output, durations = model(input_ids)
    assert isinstance(mel_output, torch.Tensor)
    assert isinstance(durations, torch.Tensor)
    assert mel_output.dim() == 3
    assert durations.shape == (batch_size, seq_len)

def test_attention_weights(model):
    """Test attention mechanism"""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Get attention weights from encoder
    encoder_output = model.encoder(input_ids)
    attention_weights = model.encoder.self_attention.attention_weights
    
    assert attention_weights.shape == (batch_size, model.config.n_heads, seq_len, seq_len)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(batch_size, model.config.n_heads, seq_len))

def test_gradient_flow(model):
    """Test gradient flow through the model"""
    model.train()
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    mel_output, durations = model(input_ids)
    
    # Compute loss
    loss = mel_output.mean() + durations.mean()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

def test_model_save_load(model, tmp_path):
    """Test model serialization"""
    # Save model
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load model
    new_model = KenyanSwahiliTTS(model.config)
    new_model.load_state_dict(torch.load(save_path))
    
    # Compare outputs
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    with torch.no_grad():
        output1 = model(input_ids)
        output2 = new_model(input_ids)
        
    assert torch.allclose(output1[0], output2[0])
    assert torch.allclose(output1[1], output2[1])

def test_input_validation(model):
    """Test input validation and error handling"""
    with pytest.raises(ValueError):
        # Empty input
        model(torch.tensor([]))
    
    with pytest.raises(ValueError):
        # Wrong input dimension
        model(torch.randn(10))
    
    with pytest.raises(ValueError):
        # Input values out of vocabulary range
        model(torch.tensor([[model.config.vocab_size + 1]]))

def test_inference_mode(model):
    """Test model behavior in inference mode"""
    model.eval()
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    with torch.no_grad():
        # Multiple forward passes should be deterministic
        output1 = model(input_ids)
        output2 = model(input_ids)
        
        assert torch.allclose(output1[0], output2[0])
        assert torch.allclose(output1[1], output2[1])
