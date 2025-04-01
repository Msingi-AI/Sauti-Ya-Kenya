"""
Preprocess audio files and generate mel spectrograms
"""
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from src.preprocessor import AudioPreprocessor

def main():
    # Setup paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    recordings_dir = data_dir / "recordings"
    
    # Load metadata
    metadata_file = processed_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_file)
    print(f"Loaded metadata with {len(metadata)} samples")
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Process each recording
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        speaker_id = row['speaker_id']
        clip_id = row['clip_id']
        wav_path = recordings_dir / f"{speaker_id}_{row['original_path']}"
        
        # Create output paths
        speaker_dir = processed_dir / speaker_id
        out_wav = speaker_dir / f"{clip_id}.wav"
        out_mel = speaker_dir / f"{clip_id}_mel.pt"
        
        # Skip if already processed
        if out_mel.exists():
            continue
            
        try:
            # Copy wav file
            if not out_wav.exists():
                os.system(f'copy "{wav_path}" "{out_wav}"')
            
            # Generate mel spectrogram
            mel = preprocessor.process_audio(str(wav_path))
            torch.save(mel, out_mel)
            
        except Exception as e:
            print(f"\nError processing {wav_path}: {e}")

if __name__ == "__main__":
    main()
