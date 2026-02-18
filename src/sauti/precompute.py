import os
import torch
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .data import get_waxal_swahili

logger = logging.getLogger(__name__)

def precompute_teacher_activations(dataset_name, config_name, out_dir, max_items=2000):
    """
    Runs the Teacher Model (Fish Speech) on WAXAL audio/text 
    and saves the hidden states to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Teacher (Fish Speech 1.5)
    TEACHER_ID = "fishaudio/fish-speech-1.5"
    logger.info(f"Loading Teacher: {TEACHER_ID}")
    
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
    teacher = AutoModel.from_pretrained(TEACHER_ID, trust_remote_code=True).to(device)
    teacher.eval()

    # 2. Load Data
    ds = get_waxal_swahili(split="train", streaming=True)
    
    # 3. Processing Loop
    logger.info(f"Processing {max_items} items...")
    count = 0
    
    with torch.no_grad():
        for i, sample in tqdm(enumerate(ds)):
            if count >= max_items:
                break
                
            text = sample['text']
            uid = f"sample_{i}"
            save_path = os.path.join(out_dir, f"{uid}.pt")
            
            # Skip if already exists (resume capability)
            if os.path.exists(save_path):
                continue

            # Tokenize & Forward Pass
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = teacher(**inputs, output_hidden_states=True)
            
            # Extract the specific layer we want to distill (e.g., last layer)
            # Fish Speech specific: check architecture for correct layer
            hidden_states = outputs.last_hidden_state.cpu()
            
            # Save
            torch.save({
                "hidden_states": hidden_states,
                "text": text,
                "audio_path": sample['audio'].get('path', ''), # Metadata
            }, save_path)
            
            count += 1

    logger.info(f"Precomputation complete. Saved {count} files to {out_dir}")