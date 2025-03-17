"""
Evaluation metrics for TTS model
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from pesq import pesq
from pystoi import stoi
import librosa

class TTSEvaluator:
    """Evaluator for TTS model performance"""
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def compute_mel_cepstral_distortion(self,
                                      predicted: torch.Tensor,
                                      target: torch.Tensor) -> float:
        """
        Compute Mel Cepstral Distortion (MCD)
        Args:
            predicted: Predicted mel-spectrogram
            target: Target mel-spectrogram
        Returns:
            MCD score (lower is better)
        """
        # Convert to log scale
        predicted_db = torch.log10(torch.clamp(predicted, min=1e-5))
        target_db = torch.log10(torch.clamp(target, min=1e-5))
        
        # Compute MCD
        diff = predicted_db - target_db
        mcd = torch.sqrt(torch.mean(diff ** 2, dim=-1))
        return mcd.mean().item()
    
    def compute_pesq(self,
                    predicted: torch.Tensor,
                    target: torch.Tensor) -> float:
        """
        Compute Perceptual Evaluation of Speech Quality (PESQ)
        Args:
            predicted: Predicted audio waveform
            target: Target audio waveform
        Returns:
            PESQ score (higher is better)
        """
        # Convert to numpy and reshape
        predicted = predicted.cpu().numpy().squeeze()
        target = target.cpu().numpy().squeeze()
        
        # Ensure same length
        min_len = min(len(predicted), len(target))
        predicted = predicted[:min_len]
        target = target[:min_len]
        
        # Compute PESQ
        try:
            score = pesq(self.sample_rate, target, predicted, 'wb')
        except:
            score = float('nan')
        
        return score
    
    def compute_stoi(self,
                    predicted: torch.Tensor,
                    target: torch.Tensor) -> float:
        """
        Compute Short-Time Objective Intelligibility (STOI)
        Args:
            predicted: Predicted audio waveform
            target: Target audio waveform
        Returns:
            STOI score (higher is better)
        """
        # Convert to numpy and reshape
        predicted = predicted.cpu().numpy().squeeze()
        target = target.cpu().numpy().squeeze()
        
        # Ensure same length
        min_len = min(len(predicted), len(target))
        predicted = predicted[:min_len]
        target = target[:min_len]
        
        # Compute STOI
        try:
            score = stoi(target, predicted, self.sample_rate, extended=False)
        except:
            score = float('nan')
        
        return score
    
    def compute_pitch_accuracy(self,
                             predicted: torch.Tensor,
                             target: torch.Tensor) -> float:
        """
        Compute pitch accuracy between predicted and target audio
        Args:
            predicted: Predicted audio waveform
            target: Target audio waveform
        Returns:
            Pitch accuracy score (higher is better)
        """
        # Convert to numpy
        predicted = predicted.cpu().numpy().squeeze()
        target = target.cpu().numpy().squeeze()
        
        # Extract pitch
        f0_predicted, voiced_flag_predicted = librosa.pyin(
            predicted,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        f0_target, voiced_flag_target = librosa.pyin(
            target,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Compute accuracy only for voiced frames
        voiced_mask = voiced_flag_predicted & voiced_flag_target
        if not np.any(voiced_mask):
            return float('nan')
            
        f0_predicted = f0_predicted[voiced_mask]
        f0_target = f0_target[voiced_mask]
        
        # Convert to cents for comparison
        cents_predicted = 1200 * np.log2(f0_predicted / 440.0)
        cents_target = 1200 * np.log2(f0_target / 440.0)
        
        # Consider a pitch correct if within 50 cents
        correct = np.abs(cents_predicted - cents_target) < 50
        return np.mean(correct)
    
    def compute_duration_accuracy(self,
                                predicted_durations: torch.Tensor,
                                target_durations: torch.Tensor) -> float:
        """
        Compute duration prediction accuracy
        Args:
            predicted_durations: Predicted phoneme durations
            target_durations: Target phoneme durations
        Returns:
            Duration accuracy score (higher is better)
        """
        # Convert to frames
        pred_frames = predicted_durations.round().long()
        target_frames = target_durations.round().long()
        
        # Compute accuracy with 20% tolerance
        tolerance = target_frames * 0.2
        correct = torch.abs(pred_frames - target_frames) <= tolerance
        return correct.float().mean().item()
    
    def evaluate_batch(self,
                      predicted_audio: torch.Tensor,
                      target_audio: torch.Tensor,
                      predicted_mel: torch.Tensor,
                      target_mel: torch.Tensor,
                      predicted_durations: torch.Tensor,
                      target_durations: torch.Tensor) -> Dict[str, float]:
        """
        Compute all evaluation metrics for a batch
        Args:
            predicted_audio: Predicted audio waveforms
            target_audio: Target audio waveforms
            predicted_mel: Predicted mel-spectrograms
            target_mel: Target mel-spectrograms
            predicted_durations: Predicted phoneme durations
            target_durations: Target phoneme durations
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Compute MCD
        metrics['mcd'] = self.compute_mel_cepstral_distortion(
            predicted_mel, target_mel
        )
        
        # Compute PESQ and STOI for each sample
        pesq_scores = []
        stoi_scores = []
        pitch_scores = []
        
        for pred, tgt in zip(predicted_audio, target_audio):
            pesq_scores.append(self.compute_pesq(pred, tgt))
            stoi_scores.append(self.compute_stoi(pred, tgt))
            pitch_scores.append(self.compute_pitch_accuracy(pred, tgt))
        
        metrics['pesq'] = np.nanmean(pesq_scores)
        metrics['stoi'] = np.nanmean(stoi_scores)
        metrics['pitch_accuracy'] = np.nanmean(pitch_scores)
        
        # Compute duration accuracy
        metrics['duration_accuracy'] = self.compute_duration_accuracy(
            predicted_durations, target_durations
        )
        
        return metrics
    
    def evaluate_model(self,
                      model,
                      eval_loader,
                      device: str = "cuda") -> Dict[str, float]:
        """
        Evaluate model on evaluation dataset
        Args:
            model: TTS model
            eval_loader: Evaluation data loader
            device: Device to run evaluation on
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_metrics = []
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                text_ids = batch['text_ids'].to(device)
                target_mel = batch['mel'].to(device)
                target_audio = batch['audio'].to(device)
                target_durations = batch['durations'].to(device)
                
                # Generate speech
                predicted_mel, predicted_durations = model(text_ids)
                predicted_audio = model.vocoder(predicted_mel)
                
                # Compute metrics
                metrics = self.evaluate_batch(
                    predicted_audio,
                    target_audio,
                    predicted_mel,
                    target_mel,
                    predicted_durations,
                    target_durations
                )
                
                all_metrics.append(metrics)
        
        # Average metrics
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[key] = np.nanmean(values)
        
        return final_metrics
