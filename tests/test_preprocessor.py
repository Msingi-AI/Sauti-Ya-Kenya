"""
Tests for text preprocessing functionality
"""
import pytest
import torch
from src.preprocessor import TextPreprocessor, SwahiliTokenizer

@pytest.fixture
def tokenizer():
    return SwahiliTokenizer(vocab_size=8000)

@pytest.fixture
def preprocessor(tokenizer):
    return TextPreprocessor(tokenizer)

def test_normalize_numbers(preprocessor):
    """Test number normalization"""
    tests = [
        ("12", "moja mbili"),
        ("3.5", "tatu nukta tano"),
        ("100", "moja sifuri sifuri"),
        ("2023", "mbili sifuri mbili tatu")
    ]
    
    for input_text, expected in tests:
        assert preprocessor.normalize_numbers(input_text) == expected

def test_normalize_expressions(preprocessor):
    """Test expression normalization"""
    tests = [
        ("sawa sawa", "sawa"),
        ("uko poa", "uko nzuri"),
        ("maze", "rafiki"),
        ("ndai", "gari")
    ]
    
    for input_text, expected in tests:
        assert preprocessor.expressions[input_text] == expected

def test_language_detection(preprocessor):
    """Test language detection"""
    text = "Hello! Habari yako?"
    tokens = preprocessor.process_text(text)
    assert len(tokens.languages) == 2
    assert "en" in tokens.languages
    assert "sw" in tokens.languages

def test_mixed_language_text(preprocessor):
    """Test processing of mixed language text"""
    text = "I will come kesho asubuhi"
    tokens = preprocessor.process_text(text)
    assert len(tokens.token_ids) > 0
    assert len(tokens.languages) > 0

def test_special_characters(preprocessor):
    """Test handling of special characters"""
    text = "Habari! Vipi? Mambo..."
    tokens = preprocessor.process_text(text)
    assert len(tokens.token_ids) > 0
