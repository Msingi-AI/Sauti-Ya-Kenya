"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
import json
from src.api import app

client = TestClient(app)

def test_root_endpoint():
    """Test API root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_synthesize_endpoint():
    """Test speech synthesis endpoint"""
    request_data = {
        "text": "Habari yako",
        "style": {
            "pitch_factor": 1.0,
            "speed_factor": 1.0,
            "energy_factor": 1.0,
            "emotion": "neutral"
        },
        "output_format": "wav"
    }
    
    response = client.post("/synthesize", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

def test_batch_synthesis():
    """Test batch synthesis endpoint"""
    request_data = {
        "items": [
            {
                "text": "Habari yako",
                "output_format": "wav"
            },
            {
                "text": "Niko sawa",
                "style": {"emotion": "happy"},
                "output_format": "mp3"
            }
        ]
    }
    
    response = client.post("/synthesize/batch", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "batch_id" in data
    assert "status_endpoint" in data

def test_tokenize_endpoint():
    """Test text tokenization endpoint"""
    request_data = {
        "text": "Hello! Habari yako?"
    }
    
    response = client.post("/tokenize", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "languages" in data

def test_style_presets():
    """Test style presets endpoint"""
    response = client.get("/styles/presets")
    assert response.status_code == 200
    data = response.json()
    assert "neutral" in data
    assert "happy" in data
    assert "sad" in data
    assert "angry" in data

def test_invalid_text():
    """Test handling of invalid input"""
    request_data = {
        "text": "",  # Empty text
        "output_format": "wav"
    }
    
    response = client.post("/synthesize", json=request_data)
    assert response.status_code == 422  # Validation error

def test_invalid_format():
    """Test handling of invalid audio format"""
    request_data = {
        "text": "Habari",
        "output_format": "invalid"  # Invalid format
    }
    
    response = client.post("/synthesize", json=request_data)
    assert response.status_code == 500  # Server error

def test_batch_status():
    """Test batch status endpoint"""
    # First create a batch
    batch_request = {
        "items": [
            {"text": "Test 1", "output_format": "wav"},
            {"text": "Test 2", "output_format": "wav"}
        ]
    }
    
    batch_response = client.post("/synthesize/batch", json=batch_request)
    batch_id = batch_response.json()["batch_id"]
    
    # Then check its status
    status_response = client.get(f"/status/{batch_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert "total" in status_data
    assert "completed" in status_data
    assert "results" in status_data
