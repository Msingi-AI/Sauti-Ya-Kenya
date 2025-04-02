# Sauti ya Kenya: Swahili Text-to-Speech System
## Technical Implementation Report
Date: April 2, 2025

## 1. System Architecture

### 1.1 Core Components
- **FastSpeech2 Model**: Non-autoregressive transformer-based architecture
  - Encoder: 4 FFT blocks with self-attention (2 heads)
  - Length Regulator: Duration predictor for variable-length synthesis
  - Decoder: 4 FFT blocks with self-attention (2 heads)
  - Mel-spectrogram Generator: Linear projection to 80 mel bins

- **HiFi-GAN Vocoder**: Neural vocoder for waveform synthesis
  - Multi-receptive field fusion
  - Multi-scale discriminator
  - Multi-period discriminator

### 1.2 Key Features
- Non-autoregressive generation for fast inference
- Robust sequence length handling (up to 10,000 tokens)
- Automatic duration prediction
- Memory-optimized training pipeline

## 2. Implementation Details

### 2.1 Text Processing
- SentencePiece tokenizer for Swahili text
- Vocabulary size: 8000 tokens
- Special token handling (BOS, EOS)
- Robust text normalization pipeline

### 2.2 Model Configuration
```python
FastSpeech2:
- d_model: 384
- n_layers: 4 (both encoder and decoder)
- n_heads: 2
- d_ff: 1536
- dropout: 0.1
- max_len: 10000
```

### 2.3 Training Setup
- Dataset: 1-hour experimental Swahili speech corpus
- Batch size: 8 with gradient accumulation
- Learning rate: Adaptive with warm-up
- Loss functions: 
  - MSE loss for mel-spectrograms
  - L1 loss for duration prediction

### 2.4 Memory Optimizations
- Gradient accumulation (4 steps)
- GPU memory monitoring
- CUDA cache cleanup
- Optimized PyTorch memory allocator settings
- Gradient clipping

## 3. Current Status

### 3.1 Achievements
- Successfully implemented FastSpeech2 architecture
- Integrated HiFi-GAN vocoder
- Developed robust text preprocessing for Swahili
- Implemented memory-efficient training pipeline
- Created demo inference script

### 3.2 Known Limitations
- Current audio output contains noise due to limited training data
- Model requires larger dataset for natural speech synthesis
- Duration prediction needs refinement
- Prosody control not yet implemented

## 4. Next Steps

### 4.1 Immediate Priorities
- Scale up training with 400-hour dataset
- Increase model capacity
- Implement data filtering and augmentation
- Add validation metrics

### 4.2 Future Improvements
- Multi-speaker support
- Prosody control
- Style transfer capabilities
- Real-time inference optimizations

## 5. Technical Debt & Considerations

### 5.1 Code Quality
- Comprehensive docstrings implemented
- Type hints used throughout
- Modular architecture for easy extension
- Debug logging implemented

### 5.2 Performance Metrics
Current metrics with 1-hour dataset:
- Training time: Not yet benchmarked
- Inference speed: Real-time capable
- Memory usage: ~2GB during inference
- Audio quality: Requires improvement

## 6. Dependencies
- PyTorch
- torchaudio
- sentencepiece
- sounddevice
- numpy
- scipy

## 7. Repository Structure
```
Sauti-Ya-Kenya-1/
├── src/
│   ├── model.py         # FastSpeech2 implementation
│   ├── vocoder.py       # HiFi-GAN implementation
│   ├── preprocessor.py  # Text processing
│   ├── inference.py     # Inference pipeline
│   └── train.py         # Training pipeline
├── examples/
│   └── tts_demo.py      # Demo script
├── data/
│   └── tokenizer/       # SentencePiece models
└── checkpoints/         # Model weights
```

## 8. Conclusion
The initial implementation provides a solid foundation for Swahili TTS. While current audio quality needs improvement due to limited training data, the architecture is ready for scaling up to the full 400-hour dataset. The modular design allows for easy extensions and improvements in the future.
