"""
Evaluation metrics for the TTS model.
"""
import os
import torch
import numpy as np
from typing import Dict
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from src.model import FastSpeech2

class TTSEvalDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'val'):
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.samples = []
        
        # Get all speaker directories
        speaker_dirs = [d for d in os.listdir(self.split_dir) 
                       if os.path.isdir(os.path.join(self.split_dir, d)) 
                       and d.startswith('Speaker')]
        
        # Collect all samples
        for speaker_dir in speaker_dirs:
            speaker_path = os.path.join(self.split_dir, speaker_dir)
            if os.path.exists(os.path.join(speaker_path, 'tokens.pt')):
                self.samples.append(speaker_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        
        # Load data
        text = torch.load(os.path.join(sample_dir, 'tokens.pt'))
        mel = torch.load(os.path.join(sample_dir, 'mel.pt'))  # [T, M]
        duration = torch.load(os.path.join(sample_dir, 'duration.pt'))
        
        print(f"\nLoaded sample from {sample_dir}:")
        print(f"text shape: {text.shape}")
        print(f"mel shape: {mel.shape}")
        print(f"duration shape: {duration.shape}")
        
        return text, mel, duration

def collate_fn(batch):
    """Create mini-batch tensors from a list of (text, mel, duration) tuples."""
    # Sort batch by text length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate inputs
    texts, mels, durations = zip(*batch)
    
    # Print input shapes
    print("\nBatch input shapes:")
    print(f"First text: {texts[0].shape}")
    print(f"First mel: {mels[0].shape}")
    print(f"First duration: {durations[0].shape}")
    
    # Get max lengths
    max_text_len = max(len(x) for x in texts)
    max_mel_len = max(x.size(0) for x in mels)
    
    # Pad sequences
    text_padded = torch.zeros(len(texts), max_text_len, dtype=texts[0].dtype)
    mel_padded = torch.zeros(len(mels), max_mel_len, mels[0].size(1), dtype=mels[0].dtype)  # [B, T, M]
    duration_padded = torch.zeros(len(durations), max_text_len, dtype=durations[0].dtype)
    
    # Fill padded tensors
    for i, (text, mel, duration) in enumerate(zip(texts, mels, durations)):
        text_padded[i, :len(text)] = text
        mel_padded[i, :mel.size(0), :] = mel  # Keep [T, M] format
        duration_padded[i, :len(duration)] = duration
    
    # Print output shapes
    print("\nBatch output shapes:")
    print(f"text_padded: {text_padded.shape}")
    print(f"mel_padded: {mel_padded.shape}")
    print(f"duration_padded: {duration_padded.shape}")
    
    return text_padded, mel_padded, duration_padded

def compute_mel_cepstral_distortion(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mel Cepstral Distortion between predicted and target spectrograms."""
    print(f"\nMCD input shapes:")
    print(f"predicted: {predicted.shape}")  # [B, T, M]
    print(f"target: {target.shape}")  # [B, T, M]
    
    # Ensure inputs have same shape
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")
        
    # Convert to log scale
    predicted = torch.log10(torch.clamp(predicted, min=1e-5))
    target = torch.log10(torch.clamp(target, min=1e-5))
    
    # Compute difference
    diff = predicted - target  # [B, T, M]
    
    # Average across mel dimension first
    mcd_per_frame = torch.sqrt(torch.mean(diff * diff, dim=-1))  # [B, T]
    print(f"MCD per frame shape: {mcd_per_frame.shape}")
    
    # Average across time dimension
    mcd_per_batch = mcd_per_frame.mean(dim=-1)  # [B]
    print(f"MCD per batch shape: {mcd_per_batch.shape}")
    
    # Average across batch
    mcd = mcd_per_batch.mean()  # scalar
    print(f"Final MCD: {mcd.item():.4f}")
    
    return mcd.item()

def evaluate_model(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    metrics = {
        'mel_loss': 0.0,
        'duration_loss': 0.0,
        'mcd': 0.0
    }
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            text, mel_target, duration_target = batch
            
            print("\nInput shapes:")
            print(f"text: {text.shape}")  # [B, T_text]
            print(f"mel_target: {mel_target.shape}")  # [B, T_mel, M]
            print(f"duration_target: {duration_target.shape}")  # [B, T_text]
            
            # Move to device
            device = next(model.parameters()).device
            text = text.to(device)
            mel_target = mel_target.to(device)
            duration_target = duration_target.to(device)
            
            # Forward pass
            mel_output, duration_pred = model(text, duration_target)
            
            print("\nOutput shapes:")
            print(f"mel_output: {mel_output.shape}")  # [B, T_mel, M]
            print(f"duration_pred: {duration_pred.shape}")  # [B, T_text]
            
            # Ensure mel spectrograms have same shape
            if mel_output.shape != mel_target.shape:
                print(f"\nShape mismatch between mel_output {mel_output.shape} and mel_target {mel_target.shape}")
                # Pad or truncate mel_target to match mel_output
                if mel_output.size(1) > mel_target.size(1):
                    # Pad mel_target
                    pad_len = mel_output.size(1) - mel_target.size(1)
                    mel_target = F.pad(mel_target, (0, 0, 0, pad_len))
                else:
                    # Truncate mel_target
                    mel_target = mel_target[:, :mel_output.size(1), :]
                print(f"After adjustment: mel_target {mel_target.shape}")
            
            # Compute losses
            mel_loss = F.l1_loss(mel_output, mel_target)
            duration_loss = F.mse_loss(duration_pred.float(), duration_target.float())
            mcd = compute_mel_cepstral_distortion(mel_output, mel_target)
            
            # Update metrics
            metrics['mel_loss'] += mel_loss.item()
            metrics['duration_loss'] += duration_loss.item()
            metrics['mcd'] += mcd
            n_batches += 1
    
    # Average metrics
    for k in metrics:
        metrics[k] /= n_batches
    return metrics

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    print("\nCheckpoint info:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_loss']:.4f}")
    
    print("\nInitializing model...")
    model = FastSpeech2(
        vocab_size=10000,  # Vocabulary size from trained model
        d_model=384,      # Model dimension
        n_enc_layers=4,   # Number of encoder layers
        n_dec_layers=4,   # Number of decoder layers
        n_heads=2,        # Number of attention heads
        d_ff=1536,       # Feed-forward dimension
        n_mels=80,       # Number of mel bands
        dropout=0.1,      # Dropout rate
        max_len=10000     # Maximum sequence length
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\nCreating validation dataset from {args.data_dir}...")
    val_dataset = TTSEvalDataset(args.data_dir, split='val')
    print(f"Found {len(val_dataset)} validation samples")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    print("\nEvaluating model...")
    metrics = evaluate_model(model, val_loader)
    
    print("\nEvaluation Results:")
    print("-" * 20)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
