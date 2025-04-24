import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        if x.size(1) > self.max_len:
            x = x[:, :self.max_len, :]
        
        x = x + self.pe[:, :x.size(1)]
        
        return x

class FFTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn))
        
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=10000):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=10000):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        return x

class LengthRegulator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.length_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x, duration_target=None):
        duration_pred = self.length_layer(x).squeeze(-1)
        
        if duration_target is None:
            batch_size, seq_len = duration_pred.shape
            
            base_durations = torch.ones_like(duration_pred) * 20
            
            if seq_len > 2:
                base_durations[:, 0] = 10
                base_durations[:, 1:-1] = 30
                base_durations[:, -1] = 15
            
            duration_pred = base_durations * (0.8 + 0.4 * torch.rand_like(duration_pred))
            duration_pred = torch.round(duration_pred)
        
        batch_size, max_length, channels = x.shape
        total_length = int(duration_pred.sum().item())
        expanded = torch.zeros(batch_size, total_length, channels).to(x.device)
        
        cur_pos = 0
        for i in range(max_length):
            dur = int(duration_pred[0, i].item())
            if dur > 0:
                if i < max_length - 1:
                    next_frame = x[:, min(i + 1, max_length - 1)]
                    for j in range(dur):
                        alpha = j / dur
                        expanded[:, cur_pos + j] = (1 - alpha) * x[:, i] + alpha * next_frame
                else:
                    expanded[:, cur_pos:cur_pos + dur] = x[:, i:i + 1]
                cur_pos += dur
        
        return expanded, duration_pred

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_enc_layers=4, n_dec_layers=4, n_heads=2, d_ff=1536, dropout=0.1, n_mels=80, max_len=10000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.encoder = Encoder(
            d_model=d_model,
            n_layers=n_enc_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        self.length_regulator = LengthRegulator(d_model)
        
        self.decoder = Decoder(
            d_model=d_model,
            n_layers=n_dec_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len
        )
        
        self.mel_linear = nn.Linear(d_model, n_mels)
        
    def forward(self, src, duration_target=None):
        x = self.embedding(src)
        
        x = self.encoder(x)
        
        x, duration_pred = self.length_regulator(x, duration_target)
        
        x = self.decoder(x)
        
        mel_output = self.mel_linear(x)
        mel_output = mel_output.transpose(1, 2)
        
        return mel_output

class TTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, mel_output, duration_pred, mel_target, duration_target):
        mel_loss = self.mse_loss(mel_output, mel_target) + 0.5 * self.mae_loss(mel_output, mel_target)
        duration_loss = self.mse_loss(duration_pred, duration_target)
        total_loss = mel_loss + duration_loss
        
        loss_dict = {
            'mel_loss': mel_loss.item(),
            'duration_loss': duration_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
