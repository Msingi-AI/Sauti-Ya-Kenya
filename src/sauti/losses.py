"""Loss helpers for Sauti distillation experiments.

Includes mel L1 loss and a placeholder hard-target loss for text/audio.
"""
import numpy as np
import logging
import torch

logger = logging.getLogger(__name__)


def mel_spectrogram(audio_array, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    try:
        import librosa
        S = librosa.feature.melspectrogram(y=audio_array.astype(float), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)
    except Exception:
        logger.warning("librosa not available or audio invalid; returning zeros mel")
        return np.zeros((n_mels, 10), dtype=np.float32)


def mel_l1_loss(pred_mel: torch.Tensor, target_mel: np.ndarray):
    if not isinstance(target_mel, torch.Tensor):
        target = torch.from_numpy(target_mel).to(pred_mel.device)
    else:
        target = target_mel.to(pred_mel.device)
    # Ensure shapes broadcast
    try:
        return torch.nn.functional.l1_loss(pred_mel, target)
    except Exception:
        logger.exception("mel_l1_loss failed due to shape mismatch")
        return torch.tensor(0.0, device=pred_mel.device)


def hard_text_loss(student_logits: torch.Tensor, target_ids: torch.Tensor):
    # Cross-entropy placeholder; user should provide tokenized targets
    try:
        loss_f = torch.nn.CrossEntropyLoss()
        # student_logits: (B, T, V) -> (B*T, V); target_ids: (B, T) -> (B*T)
        B, T, V = student_logits.shape
        logits = student_logits.view(B * T, V)
        targets = target_ids.view(B * T)
        return loss_f(logits, targets)
    except Exception:
        logger.warning("hard_text_loss fallback to zero (missing targets or mismatch)")
        return torch.tensor(0.0, device=student_logits.device)
