"""
Text-to-Speech synthesis pipeline combining TTS model and HiFiGAN vocoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Optional, Tuple

from .model import KenyanSwahiliTTS
from .vocoder import HiFiGAN
from .config import ModelConfig
from .preprocessor import TextPreprocessor, SwahiliTokenizer

class PitchShifter:
    """Pitch shifting using phase vocoder"""
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def shift_pitch(self, audio: torch.Tensor, pitch_factor: float) -> torch.Tensor:
        """
        Shift the pitch of audio
        Args:
            audio: (batch_size, 1, time) Input audio
            pitch_factor: Factor to shift pitch by (1.0 = no shift)
        Returns:
            (batch_size, 1, time) Pitch-shifted audio
        """
        if pitch_factor == 1.0:
            return audio
            
        # Convert to frequency domain
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute STFT
        stft = torch.stft(
            audio.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(audio.device),
            return_complex=True
        )
        
        # Apply phase vocoder
        time_steps = stft.size(1)
        phase_advance = torch.linspace(0, np.pi * hop_length, n_fft // 2 + 1, device=audio.device)
        phase = torch.angle(stft[:, 0])
        
        stft_stretched = []
        for i in range(time_steps):
            if i > 0:
                phase_increment = torch.angle(stft[:, i]) - torch.angle(stft[:, i-1]) - phase_advance
                phase_increment = phase_increment - 2 * np.pi * torch.round(phase_increment / (2 * np.pi))
                phase = phase + phase_advance + phase_increment * pitch_factor
            
            stft_stretched.append(torch.abs(stft[:, i]) * torch.exp(1j * phase))
        
        stft_stretched = torch.stack(stft_stretched, dim=1)
        
        # Inverse STFT
        audio_shifted = torch.istft(
            stft_stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(audio.device)
        )
        
        return audio_shifted.unsqueeze(1)

class VoiceStyle:
    """Voice style control parameters"""
    def __init__(self,
                 pitch_factor: float = 1.0,
                 speed_factor: float = 1.0,
                 energy_factor: float = 1.0,
                 emotion: str = "neutral"):
        """
        Initialize voice style parameters
        Args:
            pitch_factor: Voice pitch modification (1.0 = normal)
            speed_factor: Speech speed modification (1.0 = normal)
            energy_factor: Voice energy/volume modification (1.0 = normal)
            emotion: Emotion style ("neutral", "happy", "sad", "angry")
        """
        self.pitch_factor = pitch_factor
        self.speed_factor = speed_factor
        self.energy_factor = energy_factor
        self.emotion = emotion
        
        # Emotion presets
        self.emotion_presets = {
            "neutral": {
                "pitch": 1.0,
                "speed": 1.0,
                "energy": 1.0
            },
            "happy": {
                "pitch": 1.2,
                "speed": 1.1,
                "energy": 1.2
            },
            "sad": {
                "pitch": 0.9,
                "speed": 0.9,
                "energy": 0.8
            },
            "angry": {
                "pitch": 1.1,
                "speed": 1.2,
                "energy": 1.4
            }
        }
        
    def apply_emotion(self):
        """Apply emotion preset to factors"""
        if self.emotion in self.emotion_presets:
            preset = self.emotion_presets[self.emotion]
            self.pitch_factor *= preset["pitch"]
            self.speed_factor *= preset["speed"]
            self.energy_factor *= preset["energy"]

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
        self.pitch_shifter = PitchShifter(config.sample_rate)
        
        # Set to evaluation mode
        self.tts_model.eval()
        self.vocoder.eval()
    
    @torch.no_grad()
    def generate_speech(self, 
                       text: str,
                       style: Optional[VoiceStyle] = None) -> torch.Tensor:
        """
        Generate speech from text with style control
        Args:
            text: Input text (Swahili or mixed Swahili-English)
            style: Voice style parameters
        Returns:
            waveform: (batch_size, 1, samples) Generated speech audio
        """
        # Use default style if none provided
        if style is None:
            style = VoiceStyle()
        
        # Apply emotion preset if specified
        style.apply_emotion()
        
        # Process text
        tokens = self.text_processor.process_text(text)
        text_encodings = tokens.token_ids.to(self.device)
        
        # Generate mel-spectrogram
        mel_output, duration_pred = self.tts_model(text_encodings)
        
        # Adjust speed by scaling durations
        if style.speed_factor != 1.0:
            duration_pred = duration_pred * style.speed_factor
        
        # Convert mel-spectrogram to audio
        audio = self.vocoder.convert_mel_to_audio(mel_output)
        
        # Apply pitch shifting if needed
        if style.pitch_factor != 1.0:
            audio = self.pitch_shifter.shift_pitch(audio, style.pitch_factor)
        
        # Adjust energy/volume
        if style.energy_factor != 1.0:
            audio = audio * style.energy_factor
        
        return audio
    
    def create_style(self,
                    pitch_factor: float = 1.0,
                    speed_factor: float = 1.0,
                    energy_factor: float = 1.0,
                    emotion: str = "neutral") -> VoiceStyle:
        """
        Create a voice style configuration
        Args:
            pitch_factor: Voice pitch modification
            speed_factor: Speech speed modification
            energy_factor: Voice energy modification
            emotion: Emotion style
        Returns:
            VoiceStyle object
        """
        return VoiceStyle(
            pitch_factor=pitch_factor,
            speed_factor=speed_factor,
            energy_factor=energy_factor,
            emotion=emotion
        )
    
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
