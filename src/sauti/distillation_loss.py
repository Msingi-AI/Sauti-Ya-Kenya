import torch
import torch.nn as nn
import torch.nn.functional as F


class SwahiliDistillationLoss(nn.Module):
    """Dual-loss distillation: hard CE + soft KL with temperature.

    This module expects logits for student and teacher and ground-truth token ids.
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.T = float(temperature)
        self.alpha = float(alpha)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        """Compute combined loss.

        Args:
            student_logits: (B, T, V)
            teacher_logits: (B, T, V)
            labels: (B, T)
        Returns:
            scalar loss tensor
        """
        # Soft loss (KL on softened distributions). Use log_softmax for student.
        student_log_probs = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=-1)
        soft_loss = self.kl_div(student_log_probs, teacher_probs) * (self.T ** 2)

        # Hard loss (cross-entropy)
        B, S, V = student_logits.shape
        hard_loss = self.ce_loss(student_logits.view(B * S, V), labels.view(B * S))

        combined = (self.alpha * hard_loss) + ((1.0 - self.alpha) * soft_loss)
        return combined
