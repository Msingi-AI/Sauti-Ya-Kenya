"""
Preprocess audio files and generate mel spectrograms
"""
import os
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from src.preprocessor import AudioPreprocessor

def find_matching_wav(recordings_dir: Path, speaker_id: str) -> list:
    """Find all wav files for a speaker"""
    pattern = f"{speaker_id}_*.wav"
    return list(recordings_dir.glob(pattern))

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
        
        # Create output paths
        speaker_dir = processed_dir / speaker_id
        out_wav = speaker_dir / f"{clip_id}.wav"
        out_mel = speaker_dir / f"{clip_id}_mel.pt"
        
        # Skip if already processed
        if out_mel.exists() and out_wav.exists():
            continue
            
        try:
            # Find matching wav file
            wav_files = find_matching_wav(recordings_dir, speaker_id)
            if not wav_files:
                print(f"\nNo wav files found for {speaker_id}")
                continue
                
            # Use the first wav file for this speaker
            wav_path = wav_files[0]
            
            # Copy wav file
            if not out_wav.exists():
                print(f"\nCopying {wav_path} to {out_wav}")
                os.system(f'copy "{wav_path}" "{out_wav}"')
            
            # Generate mel spectrogram
            mel = preprocessor.process_audio(str(wav_path))
            torch.save(mel, out_mel)
            print(f"Generated mel spectrogram: {out_mel}")
            
        except Exception as e:
            print(f"\nError processing {speaker_id}: {e}")

if __name__ == "__main__":
    main()
