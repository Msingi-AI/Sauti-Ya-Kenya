"""
Tests for speech synthesis functionality
"""
import pytest
import torch
import numpy as np
from src.synthesis import TextToSpeech, VoiceStyle
from src.config import ModelConfig

@pytest.fixture
def config():
    return ModelConfig()

@pytest.fixture
def tts_engine(config):
    return TextToSpeech(config)

def test_voice_style_creation():
    """Test voice style parameter creation"""
    style = VoiceStyle(
        pitch_factor=1.2,
        speed_factor=0.9,
        energy_factor=1.1,
        emotion="happy"
    )
    assert style.pitch_factor == 1.2
    assert style.speed_factor == 0.9
    assert style.energy_factor == 1.1
    assert style.emotion == "happy"

def test_emotion_presets():
    """Test emotion preset application"""
    style = VoiceStyle(emotion="happy")
    style.apply_emotion()
    assert style.pitch_factor > 1.0
    assert style.speed_factor > 1.0
    assert style.energy_factor > 1.0

def test_speech_generation(tts_engine):
    """Test basic speech generation"""
    text = "Habari yako"
    audio = tts_engine.generate_speech(text)
    assert isinstance(audio, torch.Tensor)
    assert audio.dim() == 3  # (batch, channels, time)
    assert audio.size(0) == 1  # batch size
    assert audio.size(1) == 1  # mono audio

def test_style_modification(tts_engine):
    """Test speech generation with style modifications"""
    text = "Habari yako"
    style = VoiceStyle(pitch_factor=1.2)
    audio_modified = tts_engine.generate_speech(text, style=style)
    audio_normal = tts_engine.generate_speech(text)
    
    # Modified audio should be different from normal
    assert not torch.allclose(audio_modified, audio_normal)

def test_batch_processing(tts_engine):
    """Test batch processing capabilities"""
    texts = [
        "Habari yako",
        "Niko sawa",
        "Asante sana"
    ]
    
    styles = [
        VoiceStyle(emotion="happy"),
        VoiceStyle(emotion="neutral"),
        VoiceStyle(emotion="sad")
    ]
    
    for text, style in zip(texts, styles):
        audio = tts_engine.generate_speech(text, style=style)
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 3
