"""
Text and audio preprocessing utilities for Kenyan Swahili TTS
"""
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
    """Container for processed text tokens"""
    token_ids: List[int]
    text: str
    languages: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None

class SwahiliTokenizer:
    """SentencePiece-based tokenizer for Swahili text with code-switching support"""
    def __init__(self, vocab_size: int = 8000, model_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.sp_model = None
        if model_path:
            self.load(model_path)

    def train(self, texts: List[str], model_prefix: str):
        """Train the tokenizer on given texts"""
        # Write texts to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            for text in texts:
                # Duplicate short texts to meet minimum requirement
                if len(texts) < 100:
                    repeat = max(1, 100 // len(texts) + 1)
                    for _ in range(repeat):
                        f.write(text + '\n')
                else:
                    f.write(text + '\n')
            temp_path = f.name

        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input=temp_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=1.0,
            model_type='unigram',
            input_sentence_size=100000,  # Increased to handle more sentences
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

        # Load the trained model
        self.load(model_prefix + '.model')

        # Clean up temporary file
        import os
        os.unlink(temp_path)

    def load(self, model_path: str):
        """Load a trained tokenizer model"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        return self.sp_model.encode_as_ids(text)

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text"""
        if self.sp_model is None:
            raise RuntimeError("Tokenizer model not loaded. Call load() or train() first.")
        return self.sp_model.decode_ids(token_ids)

class TextPreprocessor:
    """Text preprocessing for Kenyan Swahili"""
    def __init__(self, tokenizer: Optional[SwahiliTokenizer] = None):
        self.tokenizer = tokenizer or SwahiliTokenizer()
        self._init_normalizers()

    def _init_normalizers(self):
        """Initialize text normalization rules"""
        # Number mapping
        self.number_map = {
            '0': 'sifuri', '1': 'moja', '2': 'mbili', '3': 'tatu',
            '4': 'nne', '5': 'tano', '6': 'sita', '7': 'saba',
            '8': 'nane', '9': 'tisa'
        }
        
        # Common Kenyan Swahili expressions and their standard forms
        self.expressions = {
            # Greetings
            "sasa": "hujambo",
            "mambo": "hujambo",
            "niaje": "hujambo",
            "vipi": "hujambo",
            
            # Common expressions
            "sawa sawa": "sawa",
            "poa": "nzuri",
            "fiti": "nzuri",
            "shwari": "nzuri",
            
            # Time expressions
            "saa hizi": "sasa",
            "saa ngapi": "saa gani",
            
            # Sheng expressions
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
            "nganya": "matatu",
            
            # Informal words
            "uko": "upo",
            "zii": "hapana",
            "sio": "siyo",
            "yawa": "aisee",
            
            # Modern expressions
            "kuconnect": "kuunganisha",
            "kudiscuss": "kujadili",
            "kupromote": "kukuza",
            "kudownload": "kupakua",
            "kusave": "kuhifadhi",
            
            # Social media
            "dm": "ujumbe",
            "status": "hali",
            "profile": "wasifu",
            
            # Technology
            "simu": "rununu",
            "kompyuta": "tarakilishi",
            "internet": "mtandao",
            "wifi": "mtandao",
            
            # Business
            "biashara": "biashara",
            "bei": "gharama",
            "ofa": "punguzo",
            
            # Transportation
            "boda": "pikipiki",
            "tuktuk": "bajaji",
            
            # Food and drinks
            "chai": "majani",
            "soda": "kinywaji",
            "juice": "sharubati",
            
            # Locations
            "town": "mjini",
            "shags": "mashambani",
            "mtaa": "mtaa",
            
            # Education
            "shule": "skuli",
            "masomo": "masomo",
            "exam": "mtihani",
            
            # Entertainment
            "movie": "filamu",
            "show": "tamasha",
            "game": "mchezo"
        }

    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to words"""
        words = []
        for word in text.split():
            if word.isdigit():
                # Convert each digit to word
                number_words = ' '.join(self.number_map[d] for d in word)
                words.append(number_words)
            else:
                words.append(word)
        return ' '.join(words)

    def normalize_expressions(self, text: str) -> str:
        """Normalize common expressions"""
        words = text.split()
        normalized = []
        i = 0
        while i < len(words):
            # Check for two-word expressions
            if i < len(words) - 1:
                two_words = ' '.join(words[i:i+2])
                if two_words in self.expressions:
                    normalized.append(self.expressions[two_words])
                    i += 2
                    continue
            
            # Check single word
            if words[i] in self.expressions:
                normalized.append(self.expressions[words[i]])
            else:
                normalized.append(words[i])
            i += 1
            
        return ' '.join(normalized)

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?\'\"()-]', '', text)
        
        return text.strip()

    def process_text(self, text: str) -> TextTokens:
        """
        Process input text
        Args:
            text: Input text
        Returns:
            TextTokens object containing token IDs and processed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Normalize expressions
        text = self.normalize_expressions(text)
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        return TextTokens(token_ids=token_ids, text=text)

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
        
    def process_audio(self, audio_path: str):
        """
        Process audio files for training/inference
        
        Args:
            audio_path: Path to wav file
            
        Returns:
            mel: Mel spectrogram tensor
        """
        # Load and resample if needed
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Generate mel spectrogram
        mel = self.mel_transform(waveform).squeeze(0).transpose(0, 1)  # [T, n_mels]
        
        return mel
