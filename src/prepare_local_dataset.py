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
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
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
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log1p(mel_spec)
    return mel_spec

def prepare_dataset(
    dataset_path: str,
    output_dir: str,
    clips_path: str = "clips",
    sentences_file: str = "validated_sentences.tsv",
    durations_file: str = "clip_durations.tsv"
):
    os.makedirs(output_dir, exist_ok=True)
    durations_path = os.path.join(dataset_path, durations_file)
    durations_df = pd.read_csv(durations_path, sep='\t')
    durations_df['duration'] = durations_df['duration[ms]'] / 1000.0
    durations_df['clip_id'] = durations_df['clip'].apply(lambda x: x.replace('.mp3', '').split('_')[-1])
    # Load sentences file and create a mapping from clip name to sentence
    sentences_path = os.path.join(dataset_path, sentences_file)
    sentences_df = pd.read_csv(sentences_path, sep='\t')
    # Try to find the column with the clip name and the sentence
    clip_col = 'path' if 'path' in sentences_df.columns else 'clip' if 'clip' in sentences_df.columns else None
    sentence_col = 'sentence' if 'sentence' in sentences_df.columns else None
    if clip_col is None or sentence_col is None:
        raise ValueError("Could not find clip or sentence columns in validated_sentences.tsv")
    clip_to_sentence = dict(zip(sentences_df[clip_col], sentences_df[sentence_col]))
    all_metadata = []
    for idx, row in tqdm(durations_df.iterrows(), total=len(durations_df)):
        try:
            audio_path = os.path.join(dataset_path, clips_path, row['clip'])
            if not os.path.exists(audio_path):
                continue
            waveform, sr = process_audio(audio_path)
            mel_spec = extract_mel_spectrogram(waveform, sr)
            speaker_dir = os.path.join(output_dir, f"Speaker_{idx:03d}")
            os.makedirs(speaker_dir, exist_ok=True)
            save_path = os.path.join(speaker_dir, f"clip_{idx:04d}")
            # Get real sentence transcription if available
            sentence = clip_to_sentence.get(row['clip'], row['clip_id'])
            torch.save(mel_spec, save_path + '_mel.pt')
            torchaudio.save(
                save_path + '.wav',
                waveform,
                sr,
                encoding='PCM_S',
                bits_per_sample=16
            )
            with open(save_path + '_text.txt', 'w', encoding='utf-8') as f:
                f.write(str(sentence))
            all_metadata.append({
                'speaker_id': f"Speaker_{idx:03d}",
                'clip_id': f"clip_{idx:04d}",
                'text': str(sentence),
                'duration': float(row['duration']),
                'mel_frames': mel_spec.shape[1],
                'original_path': row['clip']
            })
        except Exception as e:
            continue
    if all_metadata:
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)

def main():
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
    prepare_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        clips_path=args.clips_path,
        sentences_file=args.sentences_file,
        durations_file=args.durations_file
    )

if __name__ == '__main__':
    main()
