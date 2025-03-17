"""
FastAPI-based REST API for the TTS service
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Sauti Ya Kenya API",
    description="Kenyan Swahili Text-to-Speech API",
    version="0.1.0"
)

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    pitch: float = 1.0

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech
    """
    try:
        # Will implement TTS generation
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
