"""
Inference script for Kenyan Swahili TTS model
"""
import argparse
import torch
import torchaudio
from pathlib import Path

from preprocessor import TextPreprocessor, SwahiliTokenizer
from model import FastSpeech2
from vocoder import load_hifigan

def load_model(checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained FastSpeech2 model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with same parameters as training
    model = FastSpeech2(
        vocab_size=8000,  # SentencePiece vocab size
        d_model=384,      # Model dimension
        n_enc_layers=4,   # Encoder layers
        n_dec_layers=4,   # Decoder layers
        n_heads=2,        # Attention heads
        d_ff=1536,        # Feed-forward dimension
        n_mels=80,        # Mel spectrogram bins
        dropout=0.1,      # Dropout rate
        max_len=10000     # Maximum sequence length
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def synthesize(text: str, model: FastSpeech2, preprocessor: TextPreprocessor, 
               vocoder, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Synthesize speech from text"""
    # Preprocess text
    tokens = preprocessor.process_text(text)
    text_tensor = torch.tensor(tokens.token_ids).unsqueeze(0).to(device)  # [1, T]
    
    print(f"\nFastSpeech2 input shapes:")
    print(f"src: {text_tensor.shape}")
    
    # Generate mel spectrogram
    with torch.no_grad():
        print(f"\nEncoder input shape: {text_tensor.shape}")
        mel_output, _ = model(text_tensor)  # [1, T', 80]
        print(f"Mel output shape: {mel_output.shape}")
    
    # Generate waveform
    with torch.no_grad():
        audio = vocoder(mel_output)  # [1, T'']
        print(f"Audio output shape: {audio.shape}")
        
    return audio.squeeze(0).cpu()

def main():
    parser = argparse.ArgumentParser(description="Kenyan Swahili TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer and preprocessor
    tokenizer = SwahiliTokenizer()
    tokenizer.load(args.tokenizer)
    preprocessor = TextPreprocessor(tokenizer)
    
    # Load TTS model
    model = load_model(args.checkpoint, device)
    print("Loaded TTS model")
    
    # Load vocoder
    vocoder = load_hifigan(device)
    print("Loaded HiFi-GAN vocoder")
    
    # Synthesize speech
    print(f"\nSynthesizing: '{args.text}'")
    audio = synthesize(args.text, model, preprocessor, vocoder, device)
    
    # Save audio
    torchaudio.save(args.output, audio.unsqueeze(0), 22050)
    print(f"Saved audio to: {args.output}")

if __name__ == "__main__":
    main()
