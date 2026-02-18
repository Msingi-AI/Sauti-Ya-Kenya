import os
import logging
from datasets import load_dataset, Audio

logger = logging.getLogger(__name__)

def get_waxal_swahili(split="train", streaming=True):
    """
    Load the specific Swahili TTS subset from WAXAL.
    
    Args:
        split (str): 'train', 'validation', or 'test'.
        streaming (bool): If True, streams data (good for large datasets).
                          If False, downloads everything (good for caching).
    
    Returns:
        Dataset: The WAXAL Swahili dataset ready for training.
    """
    logger.info(f"Loading WAXAL (swa_tts) split={split}...")
    
    try:
        # Load specifically the 'swa_tts' config
        ds = load_dataset(
            "google/WaxalNLP", 
            "swa_tts", 
            split=split, 
            streaming=streaming, 
            trust_remote_code=True
        )
        
        # Resample audio to 16kHz (Standard for Fish Speech / CosyVoice)
        # Note: In streaming mode, casting happens on-the-fly
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        
        return ds

    except Exception as e:
        logger.error(f"Failed to load WAXAL: {e}")
        raise e

def save_to_disk(ds, output_path):
    """
    Helper to save a streaming dataset to disk (for the Volume).
    """
    logger.info(f"Saving dataset to {output_path}...")
    ds.save_to_disk(output_path)
    logger.info("Save complete.")