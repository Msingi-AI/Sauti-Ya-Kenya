"""
Test configuration and fixtures
"""
import pytest
import sys
import os
from pathlib import Path
import torch
import tempfile

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.append(str(src_dir))

@pytest.fixture
def config():
    """Create model configuration"""
    from src.config import ModelConfig
    return ModelConfig()

@pytest.fixture
def device():
    """Get compute device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def sample_texts():
    """Sample training texts"""
    return [
        "Habari yako",
        "Niko sawa",
        "Asante sana",
        "Karibu Kenya"
    ]

@pytest.fixture
def tokenizer(sample_texts):
    """Create and train tokenizer"""
    from src.preprocessor import SwahiliTokenizer
    tokenizer = SwahiliTokenizer(vocab_size=100)  # Small vocab for testing
    
    # Train tokenizer on sample texts
    with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
        model_path = f.name
    
    tokenizer.train(sample_texts, model_path[:-6])  # Remove .model extension
    return tokenizer

@pytest.fixture
def preprocessor(tokenizer):
    """Create text preprocessor"""
    from src.preprocessor import TextPreprocessor
    return TextPreprocessor(tokenizer)

@pytest.fixture
def model(config, device):
    """Create TTS model"""
    from src.model import KenyanSwahiliTTS
    return KenyanSwahiliTTS(config).to(device)

@pytest.fixture
def vocoder(config, device):
    """Create vocoder"""
    from src.vocoder import HiFiGAN
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

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after tests"""
    yield
    # Clean up any .model or .vocab files in the current directory
    for ext in ['.model', '.vocab']:
        for f in Path().glob(f'*{ext}'):
            try:
                f.unlink()
            except:
                pass
