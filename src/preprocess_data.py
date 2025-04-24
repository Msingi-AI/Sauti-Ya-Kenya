import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .preprocessor import TextPreprocessor, SwahiliTokenizer

class AudioPreprocessor:
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=1.0,
            norm="slaney",
            mel_scale="slaney"
        )

    def load_audio(self, file_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform

    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0).T  

    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    def process_audio(self, file_path: str) -> Dict[str, torch.Tensor]:
        waveform = self.load_audio(file_path)
        waveform = self.normalize_audio(waveform)
        
        mel = self.get_mel_spectrogram(waveform)
        
        duration = torch.ones(mel.size(0), dtype=torch.long)
        
        return {
            'waveform': waveform,
            'mel': mel,
            'duration': duration
        }

class DataPreprocessor:
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 val_size: float = 0.1):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.val_size = val_size
        
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_processor = AudioPreprocessor()
        self.tokenizer = SwahiliTokenizer()
        self.text_processor = TextPreprocessor(self.tokenizer)

    def load_metadata(self) -> Dict[str, Dict]:
        metadata_file = self.data_dir / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def split_data(self, metadata: Dict[str, Dict]) -> tuple[List[str], List[str]]:
        all_files = list(metadata.keys())
        np.random.shuffle(all_files)
        
        val_size = int(len(all_files) * self.val_size)
        train_files = all_files[val_size:]
        val_files = all_files[:val_size]
        
        return train_files, val_files

    def process_dataset(self):
        metadata = self.load_metadata()
        
        train_files, val_files = self.split_data(metadata)
        
        for filename in tqdm(train_files):
            self.process_example(filename, metadata[filename], self.train_dir)
            
        for filename in tqdm(val_files):
            self.process_example(filename, metadata[filename], self.val_dir)

    def process_example(self, filename: str, metadata: Dict, output_dir: Path):
        base_filename = filename.replace('.wav', '')
        
        example_dir = output_dir / base_filename
        example_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = str(self.data_dir / "recordings" / filename)
        processed_audio = self.audio_processor.process_audio(audio_path)
        
        tokens = self.text_processor.process_text(metadata["text"])
        
        mel_length = processed_audio['mel'].size(0)  
        text_length = len(tokens.text)  
        duration = torch.ones(text_length, dtype=torch.long) * (mel_length // text_length)
        remainder = mel_length % text_length
        if remainder > 0:
            duration[:remainder] += 1
        
        torch.save(processed_audio['mel'], example_dir / 'mel.pt')
        torch.save(processed_audio['waveform'], example_dir / 'waveform.pt')
        torch.save(duration, example_dir / 'duration.pt')
        torch.save(torch.tensor(tokens.token_ids), example_dir / 'tokens.pt')
        
        with open(example_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump({
                'text': metadata['text'],
                'processed_text': tokens.text,
                'speaker_id': metadata.get('speaker_id', 'unknown')
            }, f, ensure_ascii=False, indent=2)

def main():
    preprocessor = DataPreprocessor(
        data_dir='data',
        output_dir='processed_data'
    )
    preprocessor.process_dataset()

if __name__ == '__main__':
    main()
