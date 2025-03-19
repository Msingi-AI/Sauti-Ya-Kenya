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
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        checkpoint_dir,
        gradient_accumulation_steps=4
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        print("Starting training epoch...")
        progress_bar = tqdm(total=num_batches, desc="Training")
        
        self.optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(self.train_loader):
            print(f"Processing batch {batch_idx}...")
            
            # Get batch data
            text = batch['text'].to(self.device)
            mel = batch['mel'].to(self.device)
            duration = batch['duration'].to(self.device)
            
            print(f"Shapes: {text.shape} {mel.shape} {duration.shape}")
            print("Forward pass...")
            
            # Forward pass
            mel_output, duration_pred = self.model(text, duration_target=duration)
            
            # Calculate loss
            loss = self.loss_fn(mel_output, mel, duration_pred, duration)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
            # Free up memory
            del mel_output, duration_pred, loss
            torch.cuda.empty_cache()
        
        progress_bar.close()
        return {'loss': total_loss / num_batches}

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
                    loss = self.loss_fn(mel_output, mel, duration_pred, duration)
                    
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
            'loss': loss,
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            self.epochs_without_improvement = 0
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"New best model saved! Loss: {loss:.4f}")
        else:
            self.epochs_without_improvement += 1
            
        # Save periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"Saved periodic checkpoint at epoch {epoch+1}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        return checkpoint['epoch']

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
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
                print(f"No improvement for {self.max_epochs_without_improvement} epochs. Stopping training.")
                break

class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, mel_output, mel_target, duration_pred, duration_target):
        mel_loss = self.l1_loss(mel_output, mel_target)
        duration_loss = self.mse_loss(duration_pred.float(), duration_target.float())
        
        # Total loss is weighted sum
        total_loss = mel_loss + duration_loss
        
        return total_loss

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train TTS model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps')
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
    train_dataset = TTSDataset('processed_data', split='train')
    val_dataset = TTSDataset('processed_data', split='val')
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    model = FastSpeech2(
        vocab_size=10000,
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
        steps_per_epoch=len(train_loader) // args.grad_accum,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Initialize loss function
    loss_fn = TTSLoss().to(device)
    
    # Handle checkpoint directory
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.is_dir():
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            # If it exists but is a symlink, that's fine
            if not checkpoint_dir.is_symlink():
                raise
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        gradient_accumulation_steps=args.grad_accum
    )
    
    # Train model
    trainer.train(num_epochs=100, resume_from=args.resume)

if __name__ == '__main__':
    main()
