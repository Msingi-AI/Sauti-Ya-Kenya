"""
Text and audio preprocessing utilities for Kenyan Swahili TTS
"""
import re
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TextTokens:
    """Container for tokenized text with language information"""
    token_ids: torch.Tensor
    languages: List[str]  # Language tag for each token
    attention_mask: Optional[torch.Tensor] = None

class TextPreprocessor:
    """Text preprocessing for Kenyan Swahili TTS"""
    def __init__(self):
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
            "uko poa": "uko nzuri",
            "poa": "nzuri",
            "poa poa": "nzuri",
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
            
            # Kenyan Sheng expressions
            "maze": "rafiki",
            "fisi": "mtu",
            "mangware": "jioni",
            "rada": "poa",
            "kushu": "sawa",
            "matha": "mama",
            "fatha": "baba",
            "ndai": "gari",
        }
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to Swahili words"""
        def replace_number(match):
            num = match.group(0)
            if '.' in num:
                # Handle decimal numbers
                parts = num.split('.')
                return f"{' '.join(self.number_map[d] for d in parts[0])} nukta {' '.join(self.number_map[d] for d in parts[1])}"
            return ' '.join(self.number_map[d] for d in num)
        
        return re.sub(r'\d+(?:\.\d+)?', replace_number, text)
    
    def process_text(self, text: str) -> TextTokens:
        """Process input text for TTS"""
        # Normalize numbers
        text = self.normalize_numbers(text)
        
        # Simple language detection (just checking for common English words)
        words = text.split()
        languages = []
        token_ids = []
        
        english_words = {"i", "am", "is", "are", "was", "were", "will", "have", "has", "had", "the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "from", "up", "down", "come", "go", "going"}
        
        for word in words:
            word = word.lower()
            # Simple language detection
            if word in english_words:
                languages.append("en")
            else:
                languages.append("sw")
            # Simple tokenization (just using word index as token id)
            token_ids.append(len(token_ids) + 1)  # Start from 1, 0 is usually padding
        
        return TextTokens(
            token_ids=torch.tensor(token_ids),
            languages=languages
        )
