"""
Prepare Swahili TTS dataset from archive with memory optimizations
"""
import os
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import zipfile
import shutil
import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetPreparator:
    def __init__(self,
                 archive_path: str,
                 output_dir: str,
                 val_size: float = 0.1,
                 sample_rate: int = 22050,
                 n_mels: int = 80,
                 batch_size: int = 8,
                 num_workers: int = 4):
        """
        Initialize dataset preparation with memory optimizations
        Args:
            archive_path: Path to the dataset archive
            output_dir: Directory to save processed data
            val_size: Fraction of data to use for validation
            sample_rate: Target audio sample rate
            n_mels: Number of mel bands
            batch_size: Batch size for processing
            num_workers: Number of parallel workers
        """
        self.archive_path = Path(archive_path)
        self.output_dir = Path(output_dir)
        self.val_size = val_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=1.0,
            norm="slaney",
            mel_scale="slaney"
        )
        
        # Create output directories
        self.data_dir = self.output_dir / "data"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        
        for dir_path in [self.data_dir, self.train_dir, self.val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def extract_archive(self) -> Path:
        """Extract the dataset archive with progress bar"""
        self.logger.info(f"Extracting {self.archive_path}...")
        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(self.archive_path, 'r') as zip_ref:
            total_files = len(zip_ref.filelist)
            with tqdm(total=total_files, desc="Extracting") as pbar:
                for file in zip_ref.filelist:
                    zip_ref.extract(file, temp_dir)
                    pbar.update(1)
            
        return temp_dir

    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """Process a single audio file with memory cleanup"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Normalize audio
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Generate mel spectrogram
            mel = self.mel_transform(waveform)
            mel = torch.log(torch.clamp(mel, min=1e-5))
            
            result = {
                'waveform': waveform.cpu(),
                'mel': mel.squeeze(0).T.cpu()  # (time, n_mels)
            }
            
            # Clear GPU memory if used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_metadata(self, metadata_path: Path) -> Dict[str, Dict]:
        """Process metadata file with validation"""
        self.logger.info("Processing metadata...")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Validate metadata
            valid_files = {}
            for filename, info in metadata.items():
                if all(k in info for k in ['text', 'speaker_id']):
                    valid_files[filename] = info
                else:
                    self.logger.warning(f"Skipping {filename}: missing required metadata")
            
            # Split into train/val
            files = list(valid_files.keys())
            np.random.shuffle(files)
            
            val_size = int(len(files) * self.val_size)
            train_files = files[val_size:]
            val_files = files[:val_size]
            
            return {
                'train': {k: valid_files[k] for k in train_files},
                'val': {k: valid_files[k] for k in val_files}
            }
            
        except Exception as e:
            self.logger.error(f"Error processing metadata: {str(e)}")
            raise

    def process_batch(self, items: List[Tuple[str, Dict, Path]], output_dir: Path, temp_dir: Path):
        """Process a batch of examples"""
        for filename, info, example_dir in items:
            try:
                # Process audio
                audio_path = str(next(temp_dir.glob(f"**/{filename}")))
                processed_audio = self.process_audio(audio_path)
                
                if processed_audio is not None:
                    # Save processed data
                    torch.save(processed_audio['waveform'], example_dir / 'waveform.pt')
                    torch.save(processed_audio['mel'], example_dir / 'mel.pt')
                    
                    # Save metadata
                    with open(example_dir / 'metadata.json', 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
                        
                    # Clear memory
                    del processed_audio
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")

    def _process_split(self, metadata: Dict[str, Dict], output_dir: Path, temp_dir: Path):
        """Process a data split (train/val) with batching"""
        items = []
        for filename, info in metadata.items():
            example_id = filename.replace('.wav', '')
            example_dir = output_dir / example_id
            example_dir.mkdir(parents=True, exist_ok=True)
            items.append((filename, info, example_dir))
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            self.process_batch(batch, output_dir, temp_dir)
            
            # Log progress
            self.logger.info(f"Processed {i + len(batch)}/{len(items)} examples")

    def process_dataset(self):
        """Process the entire dataset with error handling"""
        temp_dir = None
        try:
            # Extract archive
            temp_dir = self.extract_archive()
            
            # Find metadata file
            try:
                metadata_file = next(temp_dir.glob("**/metadata.json"))
            except StopIteration:
                raise FileNotFoundError("metadata.json not found in archive")
            
            metadata = self.process_metadata(metadata_file)
            
            # Process training data
            self.logger.info(f"Processing {len(metadata['train'])} training examples...")
            self._process_split(metadata['train'], self.train_dir, temp_dir)
            
            # Process validation data
            self.logger.info(f"Processing {len(metadata['val'])} validation examples...")
            self._process_split(metadata['val'], self.val_dir, temp_dir)
            
            # Save dataset info
            dataset_info = {
                'num_train': len(metadata['train']),
                'num_val': len(metadata['val']),
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'speakers': self._get_speaker_info(metadata),
                'max_audio_length': self._get_max_audio_length(),
                'total_duration': self._get_total_duration(metadata)
            }
            
            with open(self.data_dir / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
                
            self.logger.info("Dataset preparation completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary files")

    def _get_speaker_info(self, metadata: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract speaker information from metadata"""
        speakers = {}
        for split_data in metadata.values():
            for info in split_data.values():
                speaker_id = info.get('speaker_id')
                if speaker_id and speaker_id not in speakers:
                    speakers[speaker_id] = {
                        'gender': info.get('gender'),
                        'age': info.get('age'),
                        'dialect': info.get('dialect'),
                        'num_utterances': 0,
                        'total_duration': 0.0
                    }
                if speaker_id:
                    speakers[speaker_id]['num_utterances'] += 1
                    speakers[speaker_id]['total_duration'] += info.get('duration', 0)
        return speakers

    def _get_max_audio_length(self) -> int:
        """Get maximum audio length in dataset"""
        max_length = 0
        for dir_path in [self.train_dir, self.val_dir]:
            for example_dir in dir_path.iterdir():
                if (example_dir / 'waveform.pt').exists():
                    waveform = torch.load(example_dir / 'waveform.pt')
                    max_length = max(max_length, waveform.size(-1))
        return max_length

    def _get_total_duration(self, metadata: Dict[str, Dict]) -> float:
        """Calculate total audio duration in hours"""
        total_duration = 0.0
        for split_data in metadata.values():
            for info in split_data.values():
                total_duration += info.get('duration', 0)
        return total_duration / 3600  # Convert to hours

def main():
    # Set memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    preparator = DatasetPreparator(
        archive_path='archive.zip',
        output_dir='processed_data',
        val_size=0.1,
        batch_size=8,
        num_workers=4
    )
    preparator.process_dataset()

if __name__ == '__main__':
    main()
