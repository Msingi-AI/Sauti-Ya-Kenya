import os
import logging
import torch
import transformers
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# Import our custom data helpers
from .data import get_waxal_swahili

logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Pads text and audio to the longest in the batch.
    """
    # 1. Extract Text & Audio
    texts = [b['text'] for b in batch]
    # WAXAL audio is already resampled to 16k in our data loader
    audios = [torch.tensor(b['audio']['array']) for b in batch] 
    
    # 2. Tokenize (we need the tokenizer passed in, usually done via partial or wrapper)
    # For simplicity here, we assume the model handles raw text or we return raw lists
    # and tokenize inside the loop (less efficient but cleaner for this script)
    return texts, audios

def finetune_student(
    model_path: str, 
    out_dir: str = "models/finetuned", 
    epochs: int = 5, 
    batch_size: int = 4, 
    learning_rate: float = 2e-5
):
    """
    Fine-tunes the distilled student model on WAXAL Swahili data.
    
    Args:
        model_path: Path to the distilled student checkpoint (or base model ID).
        out_dir: Directory to save the final Swahili model.
        epochs: Number of training passes.
    """
    # 1. Setup Accelerator (Handles Multi-GPU / Mixed Precision)
    accelerator = Accelerator(gradient_accumulation_steps=2, mixed_precision="fp16")
    logging.basicConfig(level=logging.INFO)
    logger.info(f"🚀 Initializing Sauti Fine-Tuner on {accelerator.device}")

    # 2. Load Student Model (CosyVoice 2)
    # We load with trust_remote_code=True because it's a custom architecture
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Freeze the text encoder? 
    # Usually better to keep it frozen to avoid catastrophic forgetting of language
    # for name, param in model.text_encoder.named_parameters():
    #     param.requires_grad = False

    # 3. Load Data
    logger.info("Loading WAXAL Swahili subset...")
    train_dataset = get_waxal_swahili(split="train", streaming=False) # Download for speed
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * epochs)
    )

    # 5. Prepare Everything with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 6. Training Loop
    logger.info("🔥 Starting Fine-Tuning...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process)
        
        for step, (texts, audios) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Tokenize on device
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
                
                # Audio prep (Pad audio to max length in batch)
                # Note: CosyVoice expects specific audio inputs. 
                # We assume the forward() method handles 'speech' or 'speech_values'.
                # This part depends heavily on the specific CosyVoice signature.
                # Below is a standard flow-matching signature:
                
                # Create mask and pad
                max_len = max([a.shape[0] for a in audios])
                audio_tensor = torch.zeros(len(audios), max_len).to(accelerator.device)
                for i, a in enumerate(audios):
                    audio_tensor[i, :a.shape[0]] = a.to(accelerator.device)
                
                # Forward Pass
                # Most modern TTS models return a dictionary with 'loss' if labels are provided
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    speech=audio_tensor, # Ground truth audio
                    labels=audio_tensor  # Some models use 'labels' arg for auto-loss
                )
                
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                
                # Backward
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # Save Checkpoint
        if accelerator.is_main_process:
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"✅ Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f}")
            
            save_path = os.path.join(out_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    logger.info(f"🎉 Fine-tuning complete! Model saved to {out_dir}")