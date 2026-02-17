import torch
from src.sauti.distillation_loss import SwahiliDistillationLoss


def test_swahili_distillation_loss_basic():
    torch.manual_seed(0)
    B, T, V = 2, 5, 50
    student_logits = torch.randn((B, T, V), dtype=torch.float32)
    teacher_logits = torch.randn((B, T, V), dtype=torch.float32)
    labels = torch.randint(0, V, (B, T), dtype=torch.long)

    crit = SwahiliDistillationLoss(temperature=2.0, alpha=0.6)
    loss = crit(student_logits, teacher_logits, labels)

    assert loss.dim() == 0
    assert loss.item() >= 0.0
