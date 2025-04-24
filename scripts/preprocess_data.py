import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from src.preprocessor import AudioPreprocessor

def main():
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    metadata_file = processed_dir / "metadata.csv"
    metadata = pd.read_csv(metadata_file)
    
    preprocessor = AudioPreprocessor()
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
        speaker_id = row['speaker_id']
        clip_id = row['clip_id']
        
        speaker_dir = processed_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        text_file = speaker_dir / f"{clip_id}_text.txt"
        wav_file = speaker_dir / f"{clip_id}.wav"
        mel_file = speaker_dir / f"{clip_id}_mel.pt"
        
        if text_file.exists() and wav_file.exists() and mel_file.exists():
            continue
            
        try:
            mel = preprocessor.process_audio(str(wav_file))
            torch.save(mel, mel_file)
            
        except Exception as e:
            print(f"\nError processing {speaker_id}/{clip_id}: {e}")

if __name__ == "__main__":
    main()
