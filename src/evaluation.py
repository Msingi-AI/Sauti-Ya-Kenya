"""
Evaluation metrics for the TTS model.
"""
import torch
import numpy as np
from typing import Dict, Tuple
import torchaudio
from pesq import pesq
from torch.nn import functional as F

def compute_mel_cepstral_distortion(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mel Cepstral Distortion between predicted and target spectrograms.
    
    Args:
        predicted: Predicted mel spectrogram [B, T, n_mels]
        target: Target mel spectrogram [B, T, n_mels]
    
    Returns:
        float: MCD score (lower is better)
    """
    # Convert to log scale
    predicted = torch.log10(torch.clamp(predicted, min=1e-5))
    target = torch.log10(torch.clamp(target, min=1e-5))
    
    # Compute MCD
    diff = predicted - target
    mcd = torch.sqrt(torch.mean(diff * diff, dim=-1))
    return mcd.mean().item()

def compute_pesq_score(predicted_wav: torch.Tensor, target_wav: torch.Tensor, sr: int = 22050) -> float:
    """Compute PESQ score between predicted and target waveforms.
    
    Args:
        predicted_wav: Predicted waveform [B, T]
        target_wav: Target waveform [B, T]
        sr: Sample rate
    
    Returns:
        float: PESQ score (higher is better)
    """
    # Convert to numpy
    pred_np = predicted_wav.cpu().numpy()
    target_np = target_wav.cpu().numpy()
    
    # Compute PESQ for each sample
    scores = []
    for p, t in zip(pred_np, target_np):
        try:
            score = pesq(sr, t, p, 'wb')
            scores.append(score)
        except:
            continue
    
    return np.mean(scores) if scores else 0.0

def evaluate_model(model: torch.nn.Module, 
                  val_loader: torch.utils.data.DataLoader,
                  vocoder: torch.nn.Module = None) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        model: TTS model
        val_loader: Validation data loader
        vocoder: Optional vocoder model for waveform reconstruction
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {
        'mel_loss': 0.0,
        'duration_loss': 0.0,
        'mcd': 0.0,
        'pesq': 0.0
    }
    
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text, mel_target, duration_target = batch
            
            # Move to device
            text = text.to(next(model.parameters()).device)
            mel_target = mel_target.to(next(model.parameters()).device)
            duration_target = duration_target.to(next(model.parameters()).device)
            
            # Forward pass
            mel_output, duration_pred = model(text, duration_target)
            
            # Compute losses
            mel_loss = F.l1_loss(mel_output, mel_target)
            duration_loss = F.mse_loss(duration_pred.float(), duration_target.float())
            
            # Compute MCD
            mcd = compute_mel_cepstral_distortion(mel_output, mel_target)
            
            # Compute PESQ if vocoder is available
            pesq_score = 0.0
            if vocoder is not None:
                wav_pred = vocoder(mel_output)
                wav_target = vocoder(mel_target)
                pesq_score = compute_pesq_score(wav_pred, wav_target)
            
            # Update metrics
            metrics['mel_loss'] += mel_loss.item()
            metrics['duration_loss'] += duration_loss.item()
            metrics['mcd'] += mcd
            metrics['pesq'] += pesq_score
            
            n_batches += 1
    
    # Average metrics
    for k in metrics:
        metrics[k] /= n_batches
        
    return metrics

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    from model import FastSpeech2
    from train import TTSDataset, collate_fn
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint)
    model = FastSpeech2(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create validation dataset
    val_dataset = TTSDataset(args.data_dir, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Evaluate
    metrics = evaluate_model(model, val_loader)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 20)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
