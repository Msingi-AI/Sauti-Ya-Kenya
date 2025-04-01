"""
Training script for Kenyan Swahili TTS model
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from .model import FastSpeech2, TTSLoss
from .preprocessor import TextPreprocessor, SwahiliTokenizer

class TTSDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        
        # Load metadata
        metadata_file = self.data_dir / f'{split}_metadata.csv'
        self.metadata = pd.read_csv(metadata_file)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load text tokens
        text_path = self.split_dir / 'text_tokens' / f'{row["id"]}.npy'
        text = torch.from_numpy(np.load(text_path)).long()
        
        # Load mel spectrogram
        mel_path = self.split_dir / 'mel' / f'{row["id"]}.npy'
        mel = torch.from_numpy(np.load(mel_path)).float()
        
        # Load duration
        duration_path = self.split_dir / 'duration' / f'{row["id"]}.npy'
        duration = torch.from_numpy(np.load(duration_path)).long()
        
        return text, mel, duration

def collate_fn(batch):
    """Create mini-batch tensors from a list of (text, mel, duration) tuples.
    """
    # Sort batch by text length
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate inputs
    texts, mels, durations = zip(*batch)
    
    # Get lengths
    text_lengths = [len(x) for x in texts]
    mel_lengths = [x.size(0) for x in mels]
    
    # Pad sequences
    text_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    mel_padded = pad_sequence(mels, batch_first=True, padding_value=0)
    duration_padded = pad_sequence(durations, batch_first=True, padding_value=0)
    
    return text_padded, mel_padded, duration_padded

def create_dataloader(data_dir, split='train', batch_size=32, num_workers=4, shuffle=True):
    dataset = TTSDataset(data_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, mel_output, mel_target, duration_pred, duration_target):
        """Calculate total loss.
        
        Args:
            mel_output (Tensor): Predicted mel spectrogram [B, T', M]
            mel_target (Tensor): Target mel spectrogram [B, T', M]
            duration_pred (Tensor): Predicted durations [B, T]
            duration_target (Tensor): Target durations [B, T]
            
        Returns:
            Tensor: Total loss value
        """
        # Ensure same length for mel spectrograms
        min_len = min(mel_output.size(1), mel_target.size(1))
        mel_output = mel_output[:, :min_len, :]
        mel_target = mel_target[:, :min_len, :]
        
        # Calculate losses
        mel_loss = self.l1_loss(mel_output, mel_target)
        duration_loss = self.mse_loss(duration_pred.float(), duration_target.float())
        
        # Total loss
        total_loss = mel_loss + duration_loss
        
        return total_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device,
                 checkpoint_dir, grad_accum_steps=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_accum_steps = grad_accum_steps
        self.loss_fn = TTSLoss()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        # Zero gradients at the start
        self.optimizer.zero_grad()
        
        for batch_idx, (text, mel, duration) in enumerate(tqdm(self.train_loader, desc="Training")):
            print(f"Processing batch {batch_idx}...")
            
            # Move data to device
            text = text.to(self.device)
            mel = mel.to(self.device)
            duration = duration.to(self.device)
            
            print(f"Shapes: {text.shape} {mel.shape} {duration.shape}")
            print("Forward pass...")
            
            # Forward pass
            mel_output, duration_pred = self.model(text, duration)
            
            # Calculate loss
            loss = self.loss_fn(mel_output, mel, duration_pred, duration)
            
            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if needed
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item() * self.grad_accum_steps
            
            # Clear cache
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for text, mel, duration in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                text = text.to(self.device)
                mel = mel.to(self.device)
                duration = duration.to(self.device)
                
                # Forward pass
                mel_output, duration_pred = self.model(text, duration)
                
                # Calculate loss
                loss = self.loss_fn(mel_output, mel, duration_pred, duration)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs, resume_from=None):
        start_epoch = 0
        best_loss = float('inf')
        
        # Load checkpoint if resuming
        if resume_from:
            checkpoint = torch.load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float('inf'))
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("Starting training epoch...")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss
            }
            
            # Save latest checkpoint
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pt'))
            
            # Save best checkpoint
            if is_best:
                torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pt'))
                print("Saved new best model!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train TTS model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing processed data')
    args = parser.parse_args()
    
    # Set memory optimization
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        print(f"Memory cached: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
    
    # Load tokenizer to get vocab size
    from .preprocessor import SwahiliTokenizer
    tokenizer = SwahiliTokenizer()
    tokenizer_path = os.path.join('data', 'tokenizer', 'tokenizer.model')  
    if not os.path.exists(tokenizer_path):
        raise RuntimeError(f"Tokenizer not found at {tokenizer_path}")
    tokenizer.load(tokenizer_path)
    
    # Initialize datasets
    train_loader = create_dataloader(args.data_dir, split='train', batch_size=args.batch_size, num_workers=2 if torch.cuda.is_available() else 0)
    val_loader = create_dataloader(args.data_dir, split='val', batch_size=args.batch_size, num_workers=2 if torch.cuda.is_available() else 0, shuffle=False)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    model = FastSpeech2(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_enc_layers=4,
        n_dec_layers=4,
        n_heads=2,
        d_ff=1536,
        n_mels=80,
        dropout=0.1
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.grad_accum,
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        grad_accum_steps=args.grad_accum
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)

if __name__ == '__main__':
    main()
