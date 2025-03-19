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

@dataclass
class TextTokens:
    """Container for processed text tokens"""
    token_ids: List[int]
    text: str
    languages: Optional[List[str]] = None
    attention_mask: Optional[List[int]] = None

class SwahiliTokenizer:
    """Simple tokenizer for Swahili text"""
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self._init_vocab()

    def _init_vocab(self):
        """Initialize character vocabulary"""
        # Basic Latin alphabet (lowercase)
        chars = list('abcdefghijklmnopqrstuvwxyz')
        
        # Swahili specific characters
        chars.extend(['á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'î', 'ô', 'û'])
        
        # Numbers and punctuation
        chars.extend(list('0123456789.,!?-\'\"()[] '))
        
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        
        # Create vocabulary
        for i, token in enumerate(special_tokens + chars):
            self.char_to_id[token] = i
            self.id_to_char[i] = token

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = []
        for char in text.lower():
            token_id = self.char_to_id.get(char, self.char_to_id['<unk>'])
            tokens.append(token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text"""
        return ''.join(self.id_to_char.get(id, '<unk>') for id in token_ids)

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
        
    def process_audio(self, audio_path: str):
        """
        Process audio files for training/inference
        """
        # Will implement audio preprocessing
        pass
