"""
Training script for Kenyan Swahili TTS model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.model import KenyanSwahiliTTS
from src.vocoder import HiFiGAN
from src.evaluation import TTSEvaluator

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data paths
    train_dir: str = "processed_data/train"
    val_dir: str = "processed_data/val"
    checkpoint_dir: str = "checkpoints"
    
    # Model parameters
    hidden_size: int = 384
    n_heads: int = 4
    n_layers: int = 6
    vocab_size: int = 8000
    n_mel_channels: int = 80
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    max_epochs: int = 1000
    warmup_steps: int = 4000
    grad_clip_thresh: float = 1.0
    
    # Validation
    val_interval: int = 1000
    checkpoint_interval: int = 5000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SwahiliTTSDataset(Dataset):
    """Dataset for TTS training"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.examples = []
        self._load_examples()
        
    def _load_examples(self):
        """Load all examples from data directory"""
        for example_dir in self.data_dir.iterdir():
            if example_dir.is_dir():
                self.examples.append(example_dir)
                
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example_dir = self.examples[idx]
        
        # Load tensors
        audio = torch.load(example_dir / "audio.pt")
        mel = torch.load(example_dir / "mel.pt")
        tokens = torch.load(example_dir / "tokens.pt")
        
        # Load metadata
        with open(example_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        return {
            "audio": audio,
            "mel": mel,
            "tokens": torch.tensor(tokens),
            "speaker_id": metadata["speaker_id"],
            "text": metadata["text"]
        }

class Trainer:
    """TTS model trainer"""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_wandb()
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.evaluator = TTSEvaluator()
        
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="kenyan-swahili-tts",
            config=self.config.__dict__
        )
        
    def setup_data(self):
        """Setup data loaders"""
        # Create datasets
        train_dataset = SwahiliTTSDataset(self.config.train_dir)
        val_dataset = SwahiliTTSDataset(self.config.val_dir)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    def setup_model(self):
        """Setup model, vocoder, and move to device"""
        # Create model
        self.model = KenyanSwahiliTTS(self.config)
        self.model.to(self.config.device)
        
        # Create vocoder
        self.vocoder = HiFiGAN()
        self.vocoder.to(self.config.device)
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
        
    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"model_step_{step}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["step"]
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        tokens = batch["tokens"].to(self.config.device)
        mel_target = batch["mel"].to(self.config.device)
        
        # Forward pass
        mel_output, durations = self.model(tokens)
        
        # Generate audio
        audio_output = self.vocoder(mel_output)
        
        # Compute losses
        mel_loss = nn.MSELoss()(mel_output, mel_target)
        duration_loss = nn.MSELoss()(durations, torch.ones_like(durations))  # Placeholder
        
        # Total loss
        loss = mel_loss + duration_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip_thresh
        )
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "mel_loss": mel_loss.item(),
            "duration_loss": duration_loss.item()
        }
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.model.eval()
        val_losses = []
        
        for batch in self.val_loader:
            # Move batch to device
            tokens = batch["tokens"].to(self.config.device)
            mel_target = batch["mel"].to(self.config.device)
            audio_target = batch["audio"].to(self.config.device)
            
            # Forward pass
            mel_output, durations = self.model(tokens)
            audio_output = self.vocoder(mel_output)
            
            # Compute metrics
            metrics = self.evaluator.evaluate_batch(
                audio_output,
                audio_target,
                mel_output,
                mel_target,
                durations,
                torch.ones_like(durations)  # Placeholder
            )
            
            val_losses.append(metrics)
            
        # Average metrics
        avg_metrics = {}
        for key in val_losses[0].keys():
            avg_metrics[key] = np.mean([l[key] for l in val_losses])
            
        self.model.train()
        return avg_metrics
        
    def train(self, resume_from: Optional[str] = None):
        """Train the model"""
        # Resume from checkpoint if specified
        start_step = 0
        if resume_from:
            start_step = self.load_checkpoint(resume_from)
            
        # Training loop
        step = start_step
        self.model.train()
        
        with tqdm(total=self.config.max_epochs * len(self.train_loader)) as pbar:
            for epoch in range(self.config.max_epochs):
                for batch in self.train_loader:
                    # Training step
                    losses = self.train_step(batch)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{losses['loss']:.4f}")
                    
                    # Log metrics
                    wandb.log({
                        "train/loss": losses["loss"],
                        "train/mel_loss": losses["mel_loss"],
                        "train/duration_loss": losses["duration_loss"],
                        "train/learning_rate": self.scheduler.get_last_lr()[0]
                    }, step=step)
                    
                    # Validation
                    if step > 0 and step % self.config.val_interval == 0:
                        val_metrics = self.validate()
                        wandb.log({
                            f"val/{k}": v for k, v in val_metrics.items()
                        }, step=step)
                        
                    # Save checkpoint
                    if step > 0 and step % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(step)
                        
                    step += 1
                    
        # Save final checkpoint
        self.save_checkpoint(step)
        wandb.finish()

def main():
    # Create config
    config = TrainingConfig()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
