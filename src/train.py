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
import torchaudio.transforms as T
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

from .model import FastSpeech2, TTSLoss
from .preprocessor import TextPreprocessor, SwahiliTokenizer

class TTSDataset(Dataset):
    """Dataset for TTS training"""
    def __init__(self, data_dir, metadata_path, tokenizer_path):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_path)
        
        # Initialize text preprocessor with tokenizer
        tokenizer = SwahiliTokenizer()
        tokenizer.load(tokenizer_path)
        self.preprocessor = TextPreprocessor(tokenizer)
        
        # Validate files and filter metadata
        valid_rows = []
        for idx, row in self.metadata.iterrows():
            speaker_id = row['speaker_id']
            clip_id = row['clip_id']
            
            speaker_dir = self.data_dir / speaker_id
            text_file = speaker_dir / f'{clip_id}_text.txt'
            wav_file = speaker_dir / f'{clip_id}.wav'
            mel_file = speaker_dir / f'{clip_id}_mel.pt'
            
            print(f"\nChecking files for {speaker_id}/{clip_id}:")
            print(f"Speaker dir: {speaker_dir} (exists: {speaker_dir.exists()})")
            print(f"Text file: {text_file} (exists: {text_file.exists()})")
            print(f"WAV file: {wav_file} (exists: {wav_file.exists()})")
            print(f"Mel file: {mel_file} (exists: {mel_file.exists()})")
            
            if text_file.exists() and wav_file.exists() and mel_file.exists():
                valid_rows.append(idx)
                print(" Found all required files")
            else:
                print(" Missing required files")
                
            # List contents of speaker dir if it exists
            if speaker_dir.exists():
                print(f"\nContents of {speaker_dir}:")
                for f in speaker_dir.iterdir():
                    print(f" - {f.name}")
                    
        self.metadata = self.metadata.iloc[valid_rows]
        print(f"\nFound {len(self.metadata)} valid samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a training sample"""
        row = self.metadata.iloc[idx]
        speaker_id = row['speaker_id']
        clip_id = row['clip_id']
        
        speaker_dir = self.data_dir / speaker_id
        text_file = speaker_dir / f'{clip_id}_text.txt'
        wav_file = speaker_dir / f'{clip_id}.wav'
        mel_file = speaker_dir / f'{clip_id}_mel.pt'

        # Load and tokenize text
        with open(text_file, 'r') as f:
            raw_text = f.read().strip()
            tokens = self.preprocessor.process_text(raw_text)
            text = torch.tensor(tokens.token_ids)  # Shape: [T]
            print(f"\nLoaded and tokenized text for {clip_id}: {text.shape}")
            print(f"Raw text: '{raw_text}'")
            print(f"Token IDs: {tokens.token_ids}")
        
        # Load mel spectrogram and convert to [T, n_mels]
        mel = torch.load(mel_file)  # Shape: [1, n_mels, T]
        mel = mel.squeeze(0).transpose(0, 1)  # Convert to [T, n_mels]
        print(f"Loaded and processed mel shape for {clip_id}: {mel.shape}")
        
        # Get duration from metadata
        duration = float(row['duration'])
        
        return text, mel, duration

def collate_fn(batch):
    """Collate function for dataloader that handles variable length sequences"""
    # Separate batch elements
    texts, mels, durations = zip(*batch)
    
    # Debug shapes
    print("\nText shapes in batch:")
    for i, text in enumerate(texts):
        print(f"Text {i}: {text.shape}")
    
    print("\nMel shapes in batch:")
    for i, mel in enumerate(mels):
        print(f"Mel {i}: {mel.shape}")
    
    # Get max lengths
    max_text_len = max(text.size(0) for text in texts)
    max_mel_len = max(mel.size(0) for mel in mels)
    print(f"Max text length: {max_text_len}")
    print(f"Max mel length: {max_mel_len}")
    
    # Pad texts to max length [B, T]
    text_padded = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, text in enumerate(texts):
        text_padded[i, :text.size(0)] = text
    
    # Pad mels to max length [B, T, n_mels]
    mel_padded = torch.zeros(len(mels), max_mel_len, mels[0].size(1))
    for i, mel in enumerate(mels):
        mel_padded[i, :mel.size(0), :] = mel
    
    # Convert durations to tensor and expand to match text length
    durations_expanded = torch.zeros(len(durations), max_text_len)
    for i, (text, duration) in enumerate(zip(texts, durations)):
        avg_duration = duration / text.size(0)  # Distribute duration evenly across text tokens
        durations_expanded[i, :text.size(0)] = avg_duration
    
    return text_padded, mel_padded, durations_expanded

def create_dataloader(data_dir, metadata_path, tokenizer_path, split='train', batch_size=32, num_workers=4, shuffle=True):
    dataset = TTSDataset(data_dir, metadata_path, tokenizer_path)
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
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata CSV file')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer model')
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
    
    # Initialize datasets
    train_loader = create_dataloader(args.data_dir, args.metadata_path, args.tokenizer_path, split='train', batch_size=min(args.batch_size, 4), num_workers=2 if torch.cuda.is_available() else 0)
    val_loader = create_dataloader(args.data_dir, args.metadata_path, args.tokenizer_path, split='val', batch_size=min(args.batch_size, 4), num_workers=2 if torch.cuda.is_available() else 0, shuffle=False)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {min(args.batch_size, 4)}")
    print(f"Steps per epoch: {max(1, len(train_loader) // args.grad_accum)}")
    
    # Initialize model
    model = FastSpeech2(
        vocab_size=SwahiliTokenizer().vocab_size,
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
        steps_per_epoch=max(1, len(train_loader) // args.grad_accum),  # Ensure at least 1 step
        pct_start=0.1,
        anneal_strategy='cos',
        cycle_momentum=True,
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
