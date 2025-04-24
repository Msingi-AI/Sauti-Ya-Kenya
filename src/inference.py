import argparse
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path

from preprocessor import TextPreprocessor, SwahiliTokenizer
from model import FastSpeech2
from vocoder import HiFiGAN

def load_model(checkpoint_path, device='cpu'):
    model = FastSpeech2(
        vocab_size=8000,
        d_model=384,
        n_enc_layers=4,
        n_dec_layers=4,
        n_heads=2,
        d_ff=1536,
        n_mels=80,
        dropout=0.1,
        max_len=10000
    )
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            cleaned_state_dict[k] = v
        
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            for key in missing_keys:
                if 'embedding' in key:
                    nn.init.xavier_uniform_(model.embedding.weight)
                elif 'length_layer' in key:
                    for m in model.length_regulator.length_layer.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                elif 'pe' in key:
                    pass
                else:
                    pass
        
        if unexpected_keys:
            pass
            
    except Exception as e:
        nn.init.xavier_uniform_(model.embedding.weight)
        
        for m in model.length_regulator.length_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    model = model.to(device)
    model.eval()  
    
    return model

def synthesize(text, model, preprocessor, vocoder, device='cpu'):
    tokens = preprocessor.process_text(text)
    src = torch.tensor(tokens.token_ids).unsqueeze(0).to(device)  
    
    with torch.no_grad():
        mel_output = model(src)  
    
    with torch.no_grad():
        audio = vocoder(mel_output)  
    
    return audio.squeeze(0).cpu()  

def main():
    parser = argparse.ArgumentParser(description="Kenyan Swahili TTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer model")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = SwahiliTokenizer()
    tokenizer.load(args.tokenizer)
    preprocessor = TextPreprocessor(tokenizer)
    
    model = load_model(args.checkpoint, device)
    
    vocoder = HiFiGAN(device)
    
    audio = synthesize(args.text, model, preprocessor, vocoder, device)
    
    torchaudio.save(args.output, audio.unsqueeze(0), 22050)

if __name__ == "__main__":
    main()
