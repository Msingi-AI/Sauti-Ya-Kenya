"""
Training script for Kenyan Swahili TTS model
"""
import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

from .model import KenyanSwahiliTTS
from .vocoder import HiFiGAN
from .preprocessor import TextPreprocessor, SwahiliTokenizer
from .config import ModelConfig

class SwahiliTTSDataset(Dataset):
    """Dataset for Swahili TTS training"""
    def __init__(self, data_dir: str, tokenizer: SwahiliTokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.preprocessor = TextPreprocessor(tokenizer)
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> List[Dict]:
        """Load all metadata files"""
        metadata = []
        metadata_dir = os.path.join(self.data_dir, "metadata")
        for file in os.listdir(metadata_dir):
            if file.endswith(".json"):
                with open(os.path.join(metadata_dir, file), "r") as f:
                    metadata.append(json.load(f))
        return metadata
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a training sample"""
        item = self.metadata[idx]
        
        # Load audio
        audio_path = os.path.join(self.data_dir, "wavs", item["audio_file"])
        audio, _ = load_audio(audio_path)
        
        # Process text
        text_tokens = self.preprocessor.process_text(item["text"])
        
        return {
            "text_ids": text_tokens.token_ids,
            "audio": audio,
            "speaker_id": item["speaker_id"],
            "text": item["text"]
        }

def load_audio(file_path: str) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file"""
    audio, sr = torchaudio.load(file_path)
    return audio, sr

class Trainer:
    """TTS model trainer"""
    def __init__(self,
                 config: ModelConfig,
                 train_dir: str,
                 val_dir: Optional[str] = None,
                 checkpoint_dir: str = "checkpoints",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize models
        self.tts_model = KenyanSwahiliTTS(config).to(device)
        self.vocoder = HiFiGAN(config).to(device)
        
        # Initialize tokenizer
        self.tokenizer = SwahiliTokenizer(vocab_size=config.vocab_size)
        
        # Create datasets
        self.train_dataset = SwahiliTTSDataset(train_dir, self.tokenizer)
        self.val_dataset = SwahiliTTSDataset(val_dir, self.tokenizer) if val_dir else None
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collate_fn
            )
        
        # Initialize optimizers
        self.tts_optimizer = optim.Adam(
            self.tts_model.parameters(),
            lr=config.learning_rate
        )
        self.vocoder_optimizer = optim.Adam(
            self.vocoder.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize losses
        self.mel_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()
        
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate batch samples"""
        text_ids = [item["text_ids"] for item in batch]
        audio = [item["audio"] for item in batch]
        
        # Pad sequences
        text_ids = pad_sequence(text_ids, batch_first=True)
        audio = pad_sequence(audio, batch_first=True)
        
        return {
            "text_ids": text_ids,
            "audio": audio
        }
        
    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        text_ids = batch["text_ids"].to(self.device)
        audio = batch["audio"].to(self.device)
        
        # Forward pass
        mel_output, duration_pred = self.tts_model(text_ids)
        audio_pred = self.vocoder(mel_output)
        
        # Calculate losses
        mel_loss = self.mel_loss(mel_output, audio)
        duration_loss = self.duration_loss(duration_pred, audio.size(1))
        total_loss = mel_loss + duration_loss
        
        # Backward pass
        self.tts_optimizer.zero_grad()
        self.vocoder_optimizer.zero_grad()
        total_loss.backward()
        self.tts_optimizer.step()
        self.vocoder_optimizer.step()
        
        return {
            "mel_loss": mel_loss.item(),
            "duration_loss": duration_loss.item(),
            "total_loss": total_loss.item()
        }
        
    def validate(self) -> Dict:
        """Validate model"""
        if not self.val_loader:
            return {}
            
        self.tts_model.eval()
        self.vocoder.eval()
        
        val_losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                losses = self.train_step(batch)
                val_losses.append(losses)
                
        # Average losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = sum(l[key] for l in val_losses) / len(val_losses)
            
        self.tts_model.train()
        self.vocoder.train()
        return avg_losses
        
    def save_checkpoint(self, epoch: int, losses: Dict):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "tts_model": self.tts_model.state_dict(),
            "vocoder": self.vocoder.state_dict(),
            "tts_optimizer": self.tts_optimizer.state_dict(),
            "vocoder_optimizer": self.vocoder_optimizer.state_dict(),
            "losses": losses
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.tts_model.load_state_dict(checkpoint["tts_model"])
        self.vocoder.load_state_dict(checkpoint["vocoder"])
        self.tts_optimizer.load_state_dict(checkpoint["tts_optimizer"])
        self.vocoder_optimizer.load_state_dict(checkpoint["vocoder_optimizer"])
        
        return checkpoint["epoch"]
        
    def train(self, num_epochs: int, validate_every: int = 1):
        """Train the model"""
        for epoch in range(num_epochs):
            # Training
            train_losses = []
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                losses = self.train_step(batch)
                train_losses.append(losses)
                
                # Update progress bar
                avg_loss = sum(l["total_loss"] for l in train_losses) / len(train_losses)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                val_losses = self.validate()
                
                # Log losses
                logging.info(f"Epoch {epoch+1}/{num_epochs}")
                logging.info(f"Train Loss: {avg_loss:.4f}")
                if val_losses:
                    logging.info(f"Val Loss: {val_losses['total_loss']:.4f}")
                    
                # Save checkpoint
                self.save_checkpoint(epoch + 1, {
                    "train": train_losses[-1],
                    "val": val_losses
                })

def main():
    parser = argparse.ArgumentParser(description="Train Kenyan Swahili TTS model")
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Training data directory")
    parser.add_argument("--val-dir", type=str,
                       help="Validation data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str,
                       help="Resume training from checkpoint")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    config = ModelConfig()
    trainer = Trainer(
        config=config,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logging.info(f"Resuming from epoch {start_epoch}")
    
    # Train model
    trainer.train(args.epochs - start_epoch)

if __name__ == "__main__":
    main()
