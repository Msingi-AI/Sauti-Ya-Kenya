import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SautiDistillationLoss(nn.Module):
    """
    Sauti-Specific Distillation Loss.
    
    Combines:
    1. Feature Loss (MSE): Forces Student to match Teacher's internal hidden states.
    2. Hard Loss (CE): Forces Student to predict correct WAXAL text tokens.
    3. Soft Loss (KL): Optional; only used if vocabularies are identical.
    """

    def __init__(self, temperature: float = 2.0, alpha_feat: float = 1.0, alpha_hard: float = 1.0, alpha_soft: float = 0.0):
        super().__init__()
        self.T = float(temperature)
        self.alpha_feat = float(alpha_feat)
        self.alpha_hard = float(alpha_hard)
        self.alpha_soft = float(alpha_soft)
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, 
                student_hidden: torch.Tensor, 
                teacher_hidden: torch.Tensor, 
                student_logits: torch.Tensor = None, 
                labels: torch.Tensor = None,
                teacher_logits: torch.Tensor = None):
        """
        Args:
            student_hidden: (B, T, Student_Dim) - The student's internal state
            teacher_hidden: (B, T, Student_Dim) - The teacher's state (ALREADY PROJECTED)
            student_logits: (B, T, Vocab) - Optional
            labels: (B, T) - Ground truth indices
            teacher_logits: (B, T, Vocab) - Optional
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=student_hidden.device)

        # 1. FEATURE LOSS (The Core of Sauti)
        # Robustness: Truncate to minimum sequence length if they differ slightly
        min_seq = min(student_hidden.size(1), teacher_hidden.size(1))
        
        feat_loss = self.mse_loss(
            student_hidden[:, :min_seq, :], 
            teacher_hidden[:, :min_seq, :]
        )
        total_loss += self.alpha_feat * feat_loss
        loss_dict['feat_loss'] = feat_loss.item()

        # 2. HARD LOSS (Standard Training)
        if student_logits is not None and labels is not None:
            # Flatten: (B*T, V)
            B, T, V = student_logits.shape
            # Ensure labels align with logits length
            min_len = min(T, labels.shape[1])
            
            hard_loss = self.ce_loss(
                student_logits[:, :min_len, :].reshape(-1, V), 
                labels[:, :min_len].reshape(-1)
            )
            total_loss += self.alpha_hard * hard_loss
            loss_dict['hard_loss'] = hard_loss.item()

        # 3. SOFT LOSS (Optional - only if vocabs match)
        if self.alpha_soft > 0 and teacher_logits is not None:
            if student_logits.shape[-1] != teacher_logits.shape[-1]:
                # Warn once per run (logic handled by caller usually, but safe-guard here)
                pass 
            else:
                min_seq_soft = min(student_logits.size(1), teacher_logits.size(1))
                soft_loss = self.kl_div(
                    F.log_softmax(student_logits[:, :min_seq_soft, :] / self.T, dim=-1),
                    F.softmax(teacher_logits[:, :min_seq_soft, :] / self.T, dim=-1)
                ) * (self.T ** 2)
                
                total_loss += self.alpha_soft * soft_loss
                loss_dict['soft_loss'] = soft_loss.item()

        return total_loss, loss_dict