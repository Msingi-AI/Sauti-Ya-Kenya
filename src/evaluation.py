import os
import torch
import numpy as np
from typing import Dict
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from src.model import FastSpeech2

class TTSEvalDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.samples = []
        speaker_dirs = [d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d)) and d.startswith('Speaker')]
        for speaker_dir in speaker_dirs:
            speaker_path = os.path.join(self.split_dir, speaker_dir)
            if os.path.exists(os.path.join(speaker_path, 'tokens.pt')):
                self.samples.append(speaker_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]
        text = torch.load(os.path.join(sample_dir, 'tokens.pt'))
        mel = torch.load(os.path.join(sample_dir, 'mel.pt'))
        duration = torch.load(os.path.join(sample_dir, 'duration.pt'))
        return text, mel, duration

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, durations = zip(*batch)
    max_text_len = max(len(x) for x in texts)
    max_mel_len = max(x.size(0) for x in mels)
    text_padded = torch.zeros(len(texts), max_text_len, dtype=texts[0].dtype)
    mel_padded = torch.zeros(len(mels), max_mel_len, mels[0].size(1), dtype=mels[0].dtype)
    duration_padded = torch.zeros(len(durations), max_text_len, dtype=durations[0].dtype)
    for i, (text, mel, duration) in enumerate(zip(texts, mels, durations)):
        text_padded[i, :len(text)] = text
        mel_padded[i, :mel.size(0), :] = mel
        duration_padded[i, :len(duration)] = duration
    return text_padded, mel_padded, duration_padded

def compute_mel_cepstral_distortion(predicted, target):
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: predicted {predicted.shape} vs target {target.shape}")
    predicted = torch.log10(torch.clamp(predicted, min=1e-5))
    target = torch.log10(torch.clamp(target, min=1e-5))
    diff = predicted - target
    mcd_per_frame = torch.sqrt(torch.mean(diff * diff, dim=-1))
    mcd_per_batch = mcd_per_frame.mean(dim=-1)
    mcd = mcd_per_batch.mean()
    return mcd.item()

def evaluate_model(model, val_loader):
    model.eval()
    metrics = {
        'mel_loss': 0.0,
        'duration_loss': 0.0,
        'mcd': 0.0
    }
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            text, mel_target, duration_target = batch
            device = next(model.parameters()).device
            text = text.to(device)
            mel_target = mel_target.to(device)
            duration_target = duration_target.to(device)
            mel_output, duration_pred = model(text, duration_target)
            if mel_output.shape != mel_target.shape:
                if mel_output.size(1) > mel_target.size(1):
                    pad_len = mel_output.size(1) - mel_target.size(1)
                    mel_target = F.pad(mel_target, (0, 0, 0, pad_len))
                else:
                    mel_target = mel_target[:, :mel_output.size(1), :]
            mel_loss = F.l1_loss(mel_output, mel_target)
            duration_loss = F.mse_loss(duration_pred.float(), duration_target.float())
            mcd = compute_mel_cepstral_distortion(mel_output, mel_target)
            metrics['mel_loss'] += mel_loss.item()
            metrics['duration_loss'] += duration_loss.item()
            metrics['mcd'] += mcd
            n_batches += 1
    for k in metrics:
        metrics[k] /= n_batches
    return metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = FastSpeech2(
        vocab_size=10000,
        d_model=384,
        n_enc_layers=4,
        n_dec_layers=4,
        n_heads=2,
        d_ff=1536,
        n_mels=80,
        dropout=0.1,
        max_len=10000
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    val_dataset = TTSEvalDataset(args.data_dir, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    metrics = evaluate_model(model, val_loader)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
