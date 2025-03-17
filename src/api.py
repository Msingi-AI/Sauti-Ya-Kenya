"""
FastAPI server for Kenyan Swahili TTS
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import io
import json
import logging
import torch
import torchaudio
import numpy as np
from datetime import datetime
import os

from .synthesis import TextToSpeech, VoiceStyle
from .config import ModelConfig

class TTSRequest(BaseModel):
    """Request model for text-to-speech conversion"""
    text: str
    style: Optional[Dict] = Field(
        default=None,
        description="Voice style parameters"
    )
    output_format: str = Field(
        default="wav",
        description="Audio output format (wav, mp3, ogg)"
    )

class BatchTTSRequest(BaseModel):
    """Request model for batch text-to-speech conversion"""
    items: List[TTSRequest]
    
class TokenizeRequest(BaseModel):
    """Request model for text tokenization"""
    text: str

class StylePreset(BaseModel):
    """Voice style preset parameters"""
    pitch_factor: float = 1.0
    speed_factor: float = 1.0
    energy_factor: float = 1.0
    emotion: str = "neutral"

app = FastAPI(
    title="Sauti ya Kenya API",
    description="Kenyan Swahili Text-to-Speech API",
    version="1.0.0"
)

# Initialize TTS engine
config = ModelConfig()
tts_engine = TextToSpeech(config)

@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Sauti ya Kenya - Kenyan Swahili TTS API"}

@app.post("/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text
    """
    try:
        # Create voice style if specified
        style = None
        if request.style:
            style = tts_engine.create_style(**request.style)
        
        # Generate speech
        audio = tts_engine.generate_speech(request.text, style=style)
        
        # Convert to requested format
        buffer = io.BytesIO()
        if request.output_format == "wav":
            torchaudio.save(buffer, audio, config.sample_rate, format="wav")
        elif request.output_format == "mp3":
            torchaudio.save(buffer, audio, config.sample_rate, format="mp3")
        elif request.output_format == "ogg":
            torchaudio.save(buffer, audio, config.sample_rate, format="ogg")
        else:
            raise ValueError(f"Unsupported format: {request.output_format}")
            
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type=f"audio/{request.output_format}",
            headers={
                "Content-Disposition": f'attachment; filename="audio.{request.output_format}"'
            }
        )
        
    except Exception as e:
        logging.error(f"Synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/batch")
async def synthesize_batch(
    request: BatchTTSRequest,
    background_tasks: BackgroundTasks
):
    """
    Batch synthesize speech from multiple texts
    """
    try:
        # Generate unique batch ID
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Process each item in the background
        background_tasks.add_task(
            process_batch,
            batch_id,
            request.items
        )
        
        return {
            "batch_id": batch_id,
            "message": "Batch processing started",
            "status_endpoint": f"/status/{batch_id}"
        }
        
    except Exception as e:
        logging.error(f"Batch synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Get status of batch processing
    """
    try:
        status_file = f"batch_{batch_id}_status.json"
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = json.load(f)
            return status
        else:
            raise HTTPException(status_code=404, detail="Batch not found")
            
    except Exception as e:
        logging.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    """
    Tokenize text and return token information
    """
    try:
        tokens = tts_engine.text_processor.process_text(request.text)
        return {
            "text": request.text,
            "tokens": tokens.token_ids.tolist(),
            "languages": tokens.languages
        }
        
    except Exception as e:
        logging.error(f"Tokenization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/styles/presets")
async def get_style_presets():
    """
    Get available voice style presets
    """
    return {
        "neutral": StylePreset(),
        "happy": StylePreset(
            pitch_factor=1.2,
            speed_factor=1.1,
            energy_factor=1.2,
            emotion="happy"
        ),
        "sad": StylePreset(
            pitch_factor=0.9,
            speed_factor=0.9,
            energy_factor=0.8,
            emotion="sad"
        ),
        "angry": StylePreset(
            pitch_factor=1.1,
            speed_factor=1.2,
            energy_factor=1.4,
            emotion="angry"
        )
    }

async def process_batch(batch_id: str, items: List[TTSRequest]):
    """
    Process batch synthesis request
    """
    results = []
    status = {
        "batch_id": batch_id,
        "total": len(items),
        "completed": 0,
        "failed": 0,
        "results": results
    }
    
    # Create batch directory
    batch_dir = f"batch_{batch_id}"
    os.makedirs(batch_dir, exist_ok=True)
    
    try:
        for i, item in enumerate(items):
            try:
                # Create voice style if specified
                style = None
                if item.style:
                    style = tts_engine.create_style(**item.style)
                
                # Generate speech
                audio = tts_engine.generate_speech(item.text, style=style)
                
                # Save audio file
                filename = f"audio_{i}.{item.output_format}"
                filepath = os.path.join(batch_dir, filename)
                
                torchaudio.save(
                    filepath,
                    audio,
                    config.sample_rate,
                    format=item.output_format
                )
                
                results.append({
                    "index": i,
                    "text": item.text,
                    "status": "success",
                    "file": filename
                })
                
                status["completed"] += 1
                
            except Exception as e:
                results.append({
                    "index": i,
                    "text": item.text,
                    "status": "failed",
                    "error": str(e)
                })
                
                status["failed"] += 1
                
            # Update status file
            status_file = f"batch_{batch_id}_status.json"
            with open(status_file, "w") as f:
                json.dump(status, f, indent=2)
                
    except Exception as e:
        logging.error(f"Batch processing error: {str(e)}")
        status["error"] = str(e)
        
    finally:
        # Final status update
        status["finished"] = True
        status_file = f"batch_{batch_id}_status.json"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
