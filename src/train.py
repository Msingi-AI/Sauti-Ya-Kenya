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
from typing import Dict, List, Tuple
from tqdm import tqdm

from .model import FastSpeech2, TTSLoss
from .preprocessor import TextPreprocessor, SwahiliTokenizer

class TTSDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split_dir = self.data_dir / split
        self.metadata = self._load_metadata()
        self.tokenizer = SwahiliTokenizer()
        self.text_processor = TextPreprocessor(self.tokenizer)

    def _load_metadata(self) -> List[Dict]:
        metadata = []
        for speaker_dir in self.split_dir.iterdir():
            if speaker_dir.is_dir():
                meta_file = speaker_dir / 'metadata.json'
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        speaker_meta = json.load(f)
                        metadata.append({
                            'speaker_id': speaker_dir.name,
                            'text': speaker_meta['text'],
                            'mel_path': str(speaker_dir / 'mel.pt'),
                            'duration_path': str(speaker_dir / 'duration.pt')
                        })
        return metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.metadata[idx]
        
        # Load text
        text = item['text']
        tokens = self.text_processor.process_text(text)
        
        # Load mel spectrogram
        mel = torch.load(item['mel_path'])
        
        # Load duration
        duration = torch.load(item['duration_path'])
        
        return {
            'text': tokens.token_ids,
            'mel': mel,
            'duration': duration,
            'speaker_id': item['speaker_id']
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Get max lengths
    max_text_len = max(len(x['text']) for x in batch)
    max_mel_len = max(x['mel'].size(0) for x in batch)
    
    # Initialize tensors
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), max_mel_len, 80)
    duration_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    
    # Fill tensors
    for i, item in enumerate(batch):
        text = torch.tensor(item['text'], dtype=torch.long)
        mel = item['mel']
        duration = item['duration']
        
        text_len = len(text)
        mel_len = mel.size(0)
        dur_len = len(duration)
        
        text_padded[i, :text_len] = text
        mel_padded[i, :mel_len] = mel
        duration_padded[i, :dur_len] = duration
    
    return {
        'text': text_padded,
        'mel': mel_padded,
        'duration': duration_padded
    }

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_fn: nn.Module,
                 device: torch.device,
                 checkpoint_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.max_epochs_without_improvement = 10

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0
        
        print("Starting training epoch...")
        with tqdm(self.train_loader, desc='Training') as pbar:
            print("Iterating over batches...")
            for batch_idx, batch in enumerate(pbar):
                print(f"Processing batch {batch_idx}...")
                # Move batch to device
                text = batch['text'].to(self.device)
                mel = batch['mel'].to(self.device)
                duration = batch['duration'].to(self.device)
                
                print("Shapes:", text.shape, mel.shape, duration.shape)
                
                # Forward pass
                print("Forward pass...")
                mel_output, duration_pred = self.model(text, duration_target=duration)
                
                # Calculate loss
                print("Calculating loss...")
                loss, metrics = self.loss_fn(mel_output, duration_pred, mel, duration)
                
                # Backward pass
                print("Backward pass...")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        return {'loss': total_loss / len(self.train_loader)}

    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch in pbar:
                    # Move batch to device
                    text = batch['text'].to(self.device)
                    mel = batch['mel'].to(self.device)
                    duration = batch['duration'].to(self.device)
                    
                    # Forward pass
                    mel_output, duration_pred = self.model(text, duration_target=duration)
                    
                    # Calculate loss
                    loss, metrics = self.loss_fn(mel_output, duration_pred, mel, duration)
                    
                    # Update progress
                    total_loss += loss.item()
                    pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        return {'loss': total_loss / len(self.val_loader)}

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        else:
            self.epochs_without_improvement += 1

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Training phase
            train_metrics = self.train_epoch()
            print(f'Training Loss: {train_metrics["loss"]:.4f}')
            
            # Validation phase
            val_metrics = self.validate()
            print(f'Validation Loss: {val_metrics["loss"]:.4f}')
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping
            if self.epochs_without_improvement >= self.max_epochs_without_improvement:
                print('Early stopping triggered')
                break

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize datasets
    train_dataset = TTSDataset('processed_data', split='train')
    val_dataset = TTSDataset('processed_data', split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = FastSpeech2(
        vocab_size=10000,  # Update this based on your tokenizer
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
        epochs=100,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Initialize loss function
    loss_fn = TTSLoss().to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir='checkpoints'
    )
    
    # Train model
    trainer.train(num_epochs=100)

if __name__ == '__main__':
    main()
