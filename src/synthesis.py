"""
Text-to-Speech synthesis pipeline combining TTS model and HiFiGAN vocoder
"""
import torch
import torch.nn as nn
from typing import Optional

from .model import KenyanSwahiliTTS
from .vocoder import HiFiGAN
from .config import ModelConfig
from .preprocessor import TextPreprocessor, SwahiliTokenizer

class TextToSpeech:
    def __init__(self, 
                 config: ModelConfig, 
                 tokenizer_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the complete TTS pipeline
        Args:
            config: Model configuration
            tokenizer_path: Path to trained tokenizer model
            device: Device to run inference on
        """
        self.config = config
        self.device = device
        
        # Initialize tokenizer and text processor
        self.tokenizer = SwahiliTokenizer(vocab_size=config.vocab_size)
        if tokenizer_path:
            self.tokenizer.load(tokenizer_path)
        self.text_processor = TextPreprocessor(self.tokenizer)
        
        # Initialize models
        self.tts_model = KenyanSwahiliTTS(config).to(device)
        self.vocoder = HiFiGAN(config).to(device)
        
        # Set to evaluation mode
        self.tts_model.eval()
        self.vocoder.eval()
    
    @torch.no_grad()
    def generate_speech(self, 
                       text: str,
                       speed_factor: float = 1.0) -> torch.Tensor:
        """
        Generate speech from text
        Args:
            text: Input text (Swahili or mixed Swahili-English)
            speed_factor: Speech speed factor (1.0 = normal speed)
        Returns:
            waveform: (batch_size, 1, samples) Generated speech audio
        """
        # Process text
        tokens = self.text_processor.process_text(text)
        text_encodings = tokens.token_ids.to(self.device)
        
        # Generate mel-spectrogram
        mel_output, duration_pred = self.tts_model(text_encodings)
        
        # Adjust speed by scaling durations
        if speed_factor != 1.0:
            duration_pred = duration_pred * speed_factor
        
        # Convert mel-spectrogram to audio
        audio = self.vocoder.convert_mel_to_audio(mel_output)
        
        return audio
    
    def train_tokenizer(self, texts: list[str], output_path: str):
        """
        Train the Swahili tokenizer on a list of texts
        Args:
            texts: List of training texts
            output_path: Path to save the tokenizer model
        """
        self.tokenizer.train(texts, output_path)
    
    def load_checkpoints(self, 
                        tts_checkpoint: Optional[str] = None,
                        vocoder_checkpoint: Optional[str] = None):
        """
        Load model checkpoints
        Args:
            tts_checkpoint: Path to TTS model checkpoint
            vocoder_checkpoint: Path to vocoder checkpoint
        """
        if tts_checkpoint:
            state_dict = torch.load(tts_checkpoint, map_location=self.device)
            self.tts_model.load_state_dict(state_dict['model'])
            
        if vocoder_checkpoint:
            state_dict = torch.load(vocoder_checkpoint, map_location=self.device)
            self.vocoder.load_state_dict(state_dict['generator'])
