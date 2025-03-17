"""
Text and audio preprocessing utilities for Kenyan Swahili TTS
"""
import re
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from polyglot.detect import Detector
import sentencepiece as spm

@dataclass
class TextTokens:
    """Container for tokenized text with language information"""
    token_ids: torch.Tensor
    languages: List[str]  # Language tag for each token
    attention_mask: Optional[torch.Tensor] = None

class SwahiliTokenizer:
    """Custom tokenizer for Kenyan Swahili with code-switching support"""
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.sp_model = None
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
            "<sw>": 4,  # Swahili token
            "<en>": 5,  # English token
            "<mix>": 6,  # Code-switched segment
        }
        
    def train(self, texts: List[str], output_path: str):
        """Train SentencePiece tokenizer on Swahili text"""
        # Write texts to temporary file
        with open("temp_train.txt", "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input="temp_train.txt",
            model_prefix=output_path,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type="unigram",
            pad_id=self.special_tokens["<pad>"],
            unk_id=self.special_tokens["<unk>"],
            bos_id=self.special_tokens["<s>"],
            eos_id=self.special_tokens["</s>"],
            user_defined_symbols=["<sw>", "<en>", "<mix>"]
        )
        
        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{output_path}.model")
    
    def load(self, model_path: str):
        """Load trained tokenizer"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

class TextPreprocessor:
    """Text preprocessing for Kenyan Swahili TTS"""
    def __init__(self, tokenizer: SwahiliTokenizer):
        self.tokenizer = tokenizer
        self.number_map = {
            "0": "sifuri",
            "1": "moja",
            "2": "mbili",
            "3": "tatu",
            "4": "nne",
            "5": "tano",
            "6": "sita",
            "7": "saba",
            "8": "nane",
            "9": "tisa"
        }
        
        # Common Kenyan Swahili expressions and their normalized forms
        self.expressions = {
            # Greetings and responses
            "sawa": "sawa",
            "sawa sawa": "sawa",
            "uko": "uko",
            "uko sawa": "uko sawa",
            "poa": "poa",
            "poa poa": "poa",
            "mambo": "mambo",
            "mambo vipi": "mambo",
            "niaje": "habari",
            "vipi": "vipi",
            "habari": "habari",
            "habari yako": "habari",
            "asante": "asante",
            "asante sana": "asante",
            "karibu": "karibu",
            "pole": "pole",
            "pole sana": "pole",
            
            # Common phrases
            "sasa": "habari",
            "sasa hivi": "sasa",
            "hapa": "hapa",
            "pale": "pale",
            "huko": "huko",
            "kuja": "kuja",
            "njoo": "njoo",
            "tafadhali": "tafadhali",
            "samahani": "samahani",
            
            # Time expressions
            "leo": "leo",
            "jana": "jana",
            "kesho": "kesho",
            "asubuhi": "asubuhi",
            "mchana": "mchana",
            "jioni": "jioni",
            "usiku": "usiku",
            
            # Common adjectives
            "nzuri": "nzuri",
            "mbaya": "mbaya",
            "kubwa": "kubwa",
            "ndogo": "ndogo",
            "nyingi": "nyingi",
            "kidogo": "kidogo",
            
            # Common verbs
            "kwenda": "kwenda",
            "kuja": "kuja",
            "kusema": "kusema",
            "kufanya": "kufanya",
            "kuwa": "kuwa",
            
            # Kenyan Sheng expressions (with standard Swahili mappings)
            "maze": "rafiki",
            "fisi": "mtu",
            "mangware": "jioni",
            "rada": "poa",
            "kushu": "sawa",
            "matha": "mama",
            "fatha": "baba",
            "ndai": "gari",
            
            # Question words
            "nini": "nini",
            "wapi": "wapi",
            "lini": "lini",
            "nani": "nani",
            
            # Common contractions
            "siji": "sijui",
            "uko'je": "uko vipi",
            "u'do": "uko",
            
            # Emphasis and interjections
            "kabisa": "kabisa",
            "sana": "sana",
            "ati": "ati",
            "eh": "eh",
            "ehe": "ehe",
            "ai": "ai",
            
            # Location-specific terms
            "mashinani": "vijijini",
            "mtaa": "mtaa",
            "kijiji": "kijiji",
            "shags": "nyumbani",
            
            # Extended Sheng expressions
            # Greetings and responses
            "niaje": "habari",
            "sasa": "habari",
            "poa": "nzuri",
            "mos mos": "pole pole",
            "rada": "nzuri",
            "shida": "nzuri",
            
            # People and relationships
            "maze": "rafiki",
            "manze": "rafiki",
            "bro": "ndugu",
            "braza": "ndugu",
            "mshikaji": "rafiki",
            "msee": "mtu",
            "matha": "mama",
            "fatha": "baba",
            "mbuyu": "rafiki",
            "mboka": "rafiki",
            "dem": "msichana",
            "manzi": "msichana",
            "mrembo": "msichana",
            "chali": "kijana",
            
            # Places and locations
            "mtaa": "mtaa",
            "shags": "nyumbani",
            "keja": "nyumba",
            "kwangu": "nyumbani",
            "kwetu": "nyumbani",
            "mashinani": "vijijini",
            "ghetto": "mtaa",
            
            # Transportation
            "ndai": "gari",
            "nduthi": "pikipiki",
            "mat": "matatu",
            "nganya": "matatu",
            
            # Money and business
            "mshwari": "pesa",
            "doo": "pesa",
            "thao": "elfu",
            "mbao": "ishirini",
            "bob": "shilingi",
            "mangwenje": "pesa",
            
            # Time and events
            "mangware": "jioni",
            "usiku poa": "usiku mwema",
            "kesha": "kukesha",
            "ringo": "saa",
            
            # Actions and states
            "toa": "ondoka",
            "enda": "ondoka",
            "kula": "kula",
            "lamba": "kula",
            "nyonya": "kunywa",
            "kata": "ondoka",
            "zima": "lala",
            
            # Qualities and descriptions
            "poa": "nzuri",
            "fiti": "nzuri",
            "moto": "nzuri sana",
            "fresh": "nzuri",
            "sharp": "sawa",
            "proper": "nzuri",
            "kubwa": "kubwa",
            "ndogo": "ndogo",
            
            # Common phrases
            "uko na": "una",
            "ni kubwa": "ni nzuri",
            "ni poa": "ni nzuri",
            "tukutane": "tukutane",
            "tuonane": "tuonane",
            "niko na": "nina",
            
            # Modern tech terms
            "simu": "simu",
            "foni": "simu",
            "net": "mtandao",
            "wifi": "mtandao",
            
            # Emotions and feelings
            "najam": "najisikia",
            "niko high": "nina furaha",
            "niko low": "nina huzuni",
            "nimechoka": "nimechoka",
            
            # Food and drinks
            "kanyagi": "kahawa",
            "chai": "chai",
            "chipo": "viazi",
            "nyama": "nyama",
            
            # Emphasis and interjections
            "kabisa": "kabisa",
            "tu": "tu",
            "sana": "sana",
            "ati": "ati",
            "kweli": "kweli"
        }
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to Swahili words"""
        words = []
        for word in text.split():
            if word.isdigit():
                # Convert each digit to Swahili
                swahili_num = " ".join(self.number_map[d] for d in word)
                words.append(swahili_num)
            else:
                words.append(word)
        return " ".join(words)
    
    def normalize_expressions(self, text: str) -> str:
        """Normalize common Kenyan Swahili expressions"""
        for expr, norm in self.expressions.items():
            text = re.sub(rf"\b{expr}\b", norm, text, flags=re.IGNORECASE)
        return text
    
    def detect_languages(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect language segments in text
        Returns list of (text_segment, language) tuples
        """
        try:
            detector = Detector(text)
            # If confidence is low, treat as code-switched
            if detector.language.confidence < 0.8:
                return [(text, "mix")]
            
            # Use polyglot's word-level language detection
            segments = []
            current_lang = None
            current_segment = []
            
            for word in detector.words:
                lang = "sw" if word.language.code == "sw" else "en"
                if current_lang is None:
                    current_lang = lang
                    current_segment = [word.token]
                elif lang == current_lang:
                    current_segment.append(word.token)
                else:
                    segments.append((" ".join(current_segment), current_lang))
                    current_lang = lang
                    current_segment = [word.token]
            
            if current_segment:
                segments.append((" ".join(current_segment), current_lang))
            
            return segments
        except:
            # Fallback: treat as single Swahili segment
            return [(text, "sw")]
    
    def process_text(self, text: str) -> TextTokens:
        """
        Process text for TTS
        Args:
            text: Input text (Swahili or mixed Swahili-English)
        Returns:
            TextTokens object with token IDs and language tags
        """
        # Basic normalization
        text = text.lower().strip()
        text = self.normalize_numbers(text)
        text = self.normalize_expressions(text)
        
        # Detect language segments
        segments = self.detect_languages(text)
        
        # Tokenize with language tags
        all_tokens = []
        languages = []
        
        for segment, lang in segments:
            # Add language start token
            lang_token = "<sw>" if lang == "sw" else "<en>" if lang == "en" else "<mix>"
            all_tokens.extend(self.tokenizer.sp_model.encode(lang_token))
            languages.extend([lang] * len(self.tokenizer.sp_model.encode(lang_token)))
            
            # Tokenize segment
            tokens = self.tokenizer.sp_model.encode(segment)
            all_tokens.extend(tokens)
            languages.extend([lang] * len(tokens))
        
        # Convert to tensor
        token_ids = torch.tensor(all_tokens, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(token_ids)
        
        return TextTokens(
            token_ids=token_ids,
            languages=languages,
            attention_mask=attention_mask
        )

class AudioPreprocessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def process_audio(self, audio_path: str):
        """
        Process audio files for training/inference
        """
        # Will implement audio preprocessing
        pass
