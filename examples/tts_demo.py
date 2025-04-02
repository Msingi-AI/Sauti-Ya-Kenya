"""
Demo script showing how to use the Kenyan Swahili TTS model
"""
import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_dir))

import torch
import torchaudio
import sounddevice as sd
from pathlib import Path

from preprocessor import TextPreprocessor, SwahiliTokenizer
from model import FastSpeech2
from vocoder import load_hifigan
from inference import synthesize, load_model

def main():
    # Model paths - update these with your model paths
    checkpoint_path = "checkpoints/best.pt"  # FastSpeech2 model
    tokenizer_path = "data/tokenizer/tokenizer.model"  # SentencePiece tokenizer
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize models
    tokenizer = SwahiliTokenizer()
    tokenizer.load(tokenizer_path)
    preprocessor = TextPreprocessor(tokenizer)
    
    model = load_model(checkpoint_path, device)
    print("Loaded TTS model")
    
    vocoder = load_hifigan(device)
    print("Loaded HiFi-GAN vocoder")
    
    # Example text
    text = "Karibu nyumbani kwetu. Tutakuonyesha vitu vizuri."
    print(f"\nSynthesizing text: '{text}'\n")
    
    # Create output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synthesize speech
    audio = synthesize(text, model, preprocessor, vocoder, device)
    
    # Print audio stats
    print(f"\nAudio stats:")
    print(f"Shape: {audio.shape}")
    print(f"Min: {audio.min().item():.3f}")
    print(f"Max: {audio.max().item():.3f}")
    print(f"Mean: {audio.mean().item():.3f}")
    print(f"Std: {audio.std().item():.3f}")
    
    # Save audio
    output_path = output_dir / "output.wav"
    audio = audio.squeeze().cpu()  # Remove batch dimension and move to CPU
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channels dimension for stereo
    
    try:
        torchaudio.save(str(output_path), audio, 22050)
        print(f"\nSaved audio to: {output_path}")
        
        # Try to play the audio
        print("\nPlaying audio...")
        sd.play(audio.numpy().T, 22050)
        sd.wait()  # Wait until audio finishes playing
        
    except Exception as e:
        print(f"Error saving/playing audio: {str(e)}")

if __name__ == "__main__":
    main()
