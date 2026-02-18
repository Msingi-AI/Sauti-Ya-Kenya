import pytest

torch = pytest.importorskip("torch")

from src.sauti.distillation_loss import SautiDistillationLoss


def test_sauti_distillation_loss_basic_feature_and_hard():
    torch.manual_seed(0)
    bsz, seq, vocab = 2, 5, 50
    student_hidden = torch.randn((bsz, seq, 16), dtype=torch.float32)
    teacher_hidden = torch.randn((bsz, seq, 16), dtype=torch.float32)
    student_logits = torch.randn((bsz, seq, vocab), dtype=torch.float32)
    labels = torch.randint(0, vocab, (bsz, seq), dtype=torch.long)

    crit = SautiDistillationLoss(temperature=2.0, alpha_feat=1.0, alpha_hard=0.5)
    loss, loss_dict = crit(student_hidden, teacher_hidden, student_logits=student_logits, labels=labels)

    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert "feat_loss" in loss_dict
    assert "hard_loss" in loss_dict


def test_sauti_distillation_loss_truncates_mismatched_sequence_length():
    student_hidden = torch.randn((1, 7, 8), dtype=torch.float32)
    teacher_hidden = torch.randn((1, 5, 8), dtype=torch.float32)

    crit = SautiDistillationLoss(alpha_hard=0.0)
    loss, loss_dict = crit(student_hidden, teacher_hidden)

    assert loss.item() >= 0.0
    assert "feat_loss" in loss_dict
