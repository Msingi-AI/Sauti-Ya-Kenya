import re
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import sentencepiece as spm
import torchaudio
import torchaudio.transforms as T

@dataclass
class TextTokens:
    token_ids: List[int]
    text: str
    languages: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None

class SwahiliTokenizer:
    def __init__(self, vocab_size=8000, model_path=None):
        self.vocab_size = vocab_size
        self.sp_model = None
        if model_path:
            self.load(model_path)

    def train(self, texts, model_prefix):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            for text in texts:
                if len(texts) < 100:
                    repeat = max(1, 100 // len(texts) + 1)
                    for _ in range(repeat):
                        f.write(text + '\n')
                else:
                    f.write(text + '\n')
            temp_path = f.name

        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            input_sentence_size=100000,
            shuffle_input_sentence=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>'
        )

        self.load(model_prefix + '.model')

        import os
        os.unlink(temp_path)

    def load(self, model_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

    def encode(self, text):
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        return self.sp_model.encode_as_ids(text)

    def decode(self, token_ids):
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        return self.sp_model.decode_ids(token_ids)

class TextPreprocessor:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or SwahiliTokenizer()
        self._init_normalizers()

    def _init_normalizers(self):
        self.number_map = {
            '0': 'sifuri', '1': 'moja', '2': 'mbili', '3': 'tatu',
            '4': 'nne', '5': 'tano', '6': 'sita', '7': 'saba',
            '8': 'nane', '9': 'tisa'
        }
        
        self.expressions = {
            "sasa": "hujambo",
            "mambo": "hujambo",
            "niaje": "hujambo",
            "vipi": "hujambo",
            
            "sawa sawa": "sawa",
            "poa": "nzuri",
            "fiti": "nzuri",
            "shwari": "nzuri",
            
            "saa hizi": "sasa",
            "saa ngapi": "saa gani",
            
            "maze": "rafiki",
            "manze": "rafiki",
            "bro": "ndugu",
            "demu": "msichana",
            "dame": "msichana",
            "chali": "kijana",
            "fisi": "mtu",
            "ngori": "pesa",
            "doo": "pesa",
            "mshiko": "pesa",
            "ndai": "gari",
            "nduthi": "pikipiki",
            "mathree": "matatu",
            "nganya": "matatu"
        }
        
        self.BOS = "[BOS]"
        self.EOS = "[EOS]"
        self.PAD = "[PAD]"
        
    def clean_text(self, text):
        text = text.lower().strip()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^a-z0-9\s\.,]', '', text)
        
        return text
        
    def normalize_numbers(self, text):
        words = []
        for word in text.split():
            if word.isdigit():
                number_words = [self.number_map[d] for d in word]
                words.append(" ".join(number_words))
            else:
                words.append(word)
        return " ".join(words)
        
    def normalize_expressions(self, text):
        for expr, norm in self.expressions.items():
            text = re.sub(r'\b' + expr + r'\b', norm, text)
        return text
        
    def process_text(self, text):
        text = self.clean_text(text)
        text = self.normalize_numbers(text)
        text = self.normalize_expressions(text)
        
        text = f"{self.BOS} {text} {self.EOS}"
        
        token_ids = self.tokenizer.sp_model.encode_as_ids(text)
        
        return TextTokens(
            token_ids=token_ids,
            text=text
        )

class AudioPreprocessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=8000,
            normalized=True
        )
        
    def process_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        mel = self.mel_transform(waveform).squeeze(0).transpose(0, 1)
        
        return mel
