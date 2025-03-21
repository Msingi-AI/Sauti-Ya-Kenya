"""
Process local Mozilla Common Voice dataset for TTS training.
"""
import os
import torch
import torchaudio
import pandas as pd
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
import librosa
import numpy as np
import json
import csv

def process_audio(audio_path: str, target_sr: int = 22050) -> Tuple[torch.Tensor, float]:
    """Load and preprocess audio file."""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    return waveform, target_sr

def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """Extract mel spectrogram from waveform."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=1.0,
        norm="slaney",
        mel_scale="slaney"
    )
    
    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)
    
    # Convert to log scale
    mel_spec = torch.log1p(mel_spec)
    
    return mel_spec

def prepare_dataset(
    dataset_path: str,
    output_dir: str,
    clips_path: str = "clips",
    sentences_file: str = "validated_sentences.tsv",
    durations_file: str = "clip_durations.tsv"
):
    """Prepare Mozilla Common Voice dataset for training.
    
    Args:
        dataset_path: Path to the extracted Common Voice dataset
        output_dir: Directory to save processed data
        clips_path: Name of the directory containing audio clips
        sentences_file: Name of the TSV file containing validated sentences
        durations_file: Name of the TSV file containing clip durations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read durations file first since it has the actual clips
    durations_path = os.path.join(dataset_path, durations_file)
    durations_df = pd.read_csv(durations_path, sep='\t')
    print(f"\nFound {len(durations_df)} clips with durations")
    
    # Convert duration from milliseconds to seconds
    durations_df['duration'] = durations_df['duration[ms]'] / 1000.0
    
    # Extract clip IDs from filenames
    durations_df['clip_id'] = durations_df['clip'].apply(lambda x: x.replace('.mp3', '').split('_')[-1])
    
    # Process each clip
    all_metadata = []
    for idx, row in tqdm(durations_df.iterrows(), desc="Processing clips", total=len(durations_df)):
        try:
            # Load and process audio
            audio_path = os.path.join(dataset_path, clips_path, row['clip'])
            if not os.path.exists(audio_path):
                print(f"\nWarning: Audio file not found: {audio_path}")
                continue
                
            waveform, sr = process_audio(audio_path)
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(waveform, sr)
            
            # Create speaker directory (for now, treat each clip as its own speaker)
            speaker_dir = os.path.join(output_dir, f"Speaker_{idx:03d}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            # Save processed data
            save_path = os.path.join(speaker_dir, f"clip_{idx:04d}")
            torch.save(mel_spec, save_path + '_mel.pt')
            
            # Save normalized audio as WAV for augmentation
            torchaudio.save(
                save_path + '.wav',
                waveform,
                sr,
                encoding='PCM_S',
                bits_per_sample=16
            )
            
            # Save text (use clip ID as placeholder, we'll update this later)
            with open(save_path + '_text.txt', 'w', encoding='utf-8') as f:
                f.write(row['clip_id'])
            
            # Collect metadata
            all_metadata.append({
                'speaker_id': f"Speaker_{idx:03d}",
                'clip_id': f"clip_{idx:04d}",
                'text': row['clip_id'],  # Placeholder
                'duration': float(row['duration']),
                'mel_frames': mel_spec.shape[1],
                'original_path': row['clip']
            })
            
        except Exception as e:
            print(f"\nError processing clip {row['clip']}: {str(e)}")
            continue
    
    # Save metadata
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        print(f"\nProcessed {len(all_metadata)} clips successfully")
        print(f"Total duration: {sum(m['duration'] for m in all_metadata) / 3600:.2f} hours")
        print(f"Average duration: {np.mean([m['duration'] for m in all_metadata]):.2f} seconds")
    else:
        print("\nNo clips were processed successfully")

def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description="Process local Mozilla Common Voice dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to the extracted Common Voice dataset")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                      help="Directory to save processed data")
    parser.add_argument("--clips_path", type=str, default="clips",
                      help="Name of the directory containing audio clips")
    parser.add_argument("--sentences_file", type=str, default="validated_sentences.tsv",
                      help="Name of the TSV file containing validated sentences")
    parser.add_argument("--durations_file", type=str, default="clip_durations.tsv",
                      help="Name of the TSV file containing clip durations")
    
    args = parser.parse_args()
    
    print("\nProcessing dataset...")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    prepare_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        clips_path=args.clips_path,
        sentences_file=args.sentences_file,
        durations_file=args.durations_file
    )

if __name__ == '__main__':
    main()
