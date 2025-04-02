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

def load_model(checkpoint_path: str, device: str = 'cpu') -> FastSpeech2:
    """
    Load FastSpeech2 model from checkpoint
    """
    # Initialize model with correct vocab size
    model = FastSpeech2(
        vocab_size=8000,   # Match SentencePiece tokenizer vocab size
        d_model=384,
        n_enc_layers=4,    # Encoder layers
        n_dec_layers=4,    # Decoder layers
        n_heads=2,
        d_ff=1536,
        n_mels=80,
        dropout=0.1,
        max_len=10000
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
    
    # Clean up state dict
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present (from DataParallel)
        if k.startswith('module.'):
            k = k[7:]
        cleaned_state_dict[k] = v
    
    # Load cleaned state dict
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing_keys:
        print("Missing keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected keys:", unexpected_keys)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model

def synthesize(text: str, model: FastSpeech2, preprocessor: TextPreprocessor, 
               vocoder, device: str = 'cpu') -> torch.Tensor:
    """
    Synthesize speech from text
    Args:
        text: Input text
        model: FastSpeech2 model
        preprocessor: Text preprocessor
        vocoder: HiFi-GAN vocoder
        device: Device to run inference on
    Returns:
        audio: Audio waveform tensor
    """
    # Preprocess text
    print(f"\nProcessing text: '{text}'")
    tokens = preprocessor.process_text(text)
    print(f"Token IDs: {tokens.token_ids}")
    
    # Convert to tensor
    src = torch.tensor(tokens.token_ids).unsqueeze(0).to(device)  # [1, T]
    print(f"\nFastSpeech2 input shapes:")
    print(f"src: {src.shape}")
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel_output = model(src)  # [1, 80, T']
        print(f"Mel output shape: {mel_output.shape}")
    
    # Convert to audio
    with torch.no_grad():
        audio = vocoder(mel_output)  # [1, 1, T'']
        print(f"Audio output shape: {audio.shape}")
    
    return audio.squeeze(0).cpu()  # Return [1, T''] tensor

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
