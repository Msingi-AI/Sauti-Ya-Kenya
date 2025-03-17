"""
FastAPI-based REST API for the TTS service
"""
import io
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .synthesis import TextToSpeech
from .config import ModelConfig

app = FastAPI(
    title="Sauti Ya Kenya API",
    description="Kenyan Swahili Text-to-Speech API",
    version="0.1.0"
)

# Initialize TTS pipeline
config = ModelConfig()
tts_pipeline = TextToSpeech(config)

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech (Swahili or mixed Swahili-English)")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed factor (1.0 = normal speed)")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Voice pitch factor (1.0 = normal pitch)")

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech
    Returns audio file in WAV format
    """
    try:
        # TODO: Add text preprocessing and tokenization
        # For now, using dummy text encodings
        text_encodings = torch.randint(0, 100, (1, 10))  # Replace with actual text processing
        
        # Generate audio
        audio = tts_pipeline.generate_speech(
            text_encodings=text_encodings,
            speed_factor=request.speed
        )
        
        # Convert to WAV format
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), tts_pipeline.config.sample_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="generated_speech.wav"'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": tts_pipeline.device
    }
