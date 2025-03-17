"""
Test configuration and fixtures
"""
import pytest
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.append(str(src_dir))

# Import required modules
import torch
from src.config import ModelConfig
from src.model import KenyanSwahiliTTS
from src.vocoder import HiFiGAN
from src.preprocessor import TextPreprocessor, SwahiliTokenizer

@pytest.fixture
def config():
    """Create model configuration"""
    return ModelConfig()

@pytest.fixture
def device():
    """Get compute device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def tokenizer():
    """Create tokenizer"""
    return SwahiliTokenizer(vocab_size=8000)

@pytest.fixture
def preprocessor(tokenizer):
    """Create text preprocessor"""
    return TextPreprocessor(tokenizer)

@pytest.fixture
def model(config, device):
    """Create TTS model"""
    return KenyanSwahiliTTS(config).to(device)

@pytest.fixture
def vocoder(config, device):
    """Create vocoder"""
    return HiFiGAN(config).to(device)

@pytest.fixture
def sample_text():
    """Sample input text"""
    return "Habari yako"

@pytest.fixture
def sample_audio(device):
    """Sample audio tensor"""
    return torch.randn(1, 1, 16000).to(device)

@pytest.fixture
def sample_mel(device):
    """Sample mel spectrogram"""
    return torch.randn(1, 80, 100).to(device)  # (batch, n_mels, time)
