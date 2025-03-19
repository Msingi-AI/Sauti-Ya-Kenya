# 🎙️ Sauti Ya Kenya

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/sauti_ya_kenya_training.ipynb)

<img src="docs/images/logo.png" alt="Sauti Ya Kenya Logo" width="200"/>

*Bringing Kenyan Swahili voices to life through AI* 🇰🇪

[Getting Started](#getting-started) •
[Features](#features) •
[Documentation](#documentation) •
[Contributing](#contributing) •
[Contact](#contact)

</div>

---

## 🌟 Overview

Sauti Ya Kenya is a state-of-the-art Text-to-Speech (TTS) system specifically designed for Kenyan Swahili. Our model handles:
- 🗣️ Natural Kenyan Swahili pronunciation
- 🔄 Code-switching between Swahili and English
- 📢 Multiple speaker voices
- 🎯 Local expressions and idioms

## ✨ Features

- **High-Quality Speech** - FastSpeech 2 architecture for fast, high-fidelity voice synthesis
- **Code-Switching** - Seamlessly handles mixed Swahili-English text
- **Local Optimization** - Tuned specifically for Kenyan Swahili phonetics
- **Easy Integration** - Simple API for quick integration into your projects
- **Memory Efficient** - Optimized for training on Google Colab

## 🚀 Getting Started

### Prerequisites

```bash
python 3.8+
pytorch 2.0+
torchaudio
numpy
pandas
soundfile
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Msingi-AI/Sauti-Ya-Kenya.git
cd Sauti-Ya-Kenya

# Install dependencies
pip install -r requirements.txt
```

## Contributing Voice Data 🗣️

We welcome voice contributions from Kenyan Swahili speakers! Your voice will help create a more diverse and representative model.

### Speaker Requirements

- Native or fluent Kenyan Swahili speaker
- Clear pronunciation
- Comfortable with both Swahili and English (for code-switching)
- Consistent speaking style

### Recording Guidelines

1. **Environment**:
   - Find a quiet room with minimal echo
   - Avoid background noise and interruptions
   - Use a good quality microphone if possible
   - Record during quiet hours to minimize ambient noise

2. **Recording Setup**:
   - Run the data collection tool:
     ```bash
     python -m src.data_collection
     ```
   - Fill in your speaker details in the GUI:
     - Name (or pseudonym)
     - Age range
     - Gender
     - Primary dialect/region
     - Years speaking Kenyan Swahili
   - Maintain consistent distance from microphone (about 6-8 inches)
   - Speak clearly and naturally

3. **Recording Process**:
   - Each session will present text prompts to read
   - Take time to read and understand each prompt before recording
   - Click "Record" to start recording
   - Read the text naturally at your normal speaking pace
   - Feel free to use your natural code-switching style
   - Click "Stop" when finished
   - Review the recording and re-record if needed
   - Save the recording when satisfied

4. **Quality Check**:
   - Recordings should be clear and understandable
   - No background noise or interruptions
   - Natural speaking rhythm and intonation
   - Consistent volume level
   - Authentic Kenyan Swahili pronunciation

### Text Guidelines

When recording, you'll encounter different types of text:
1. Pure Swahili sentences
2. Mixed Swahili-English sentences (code-switching)
3. Common Kenyan expressions
4. Numbers and dates
5. Questions and exclamations

Read each type naturally, as you would in everyday conversation.

## Training the Model 🧠

### Local Training

1. Preprocess the data:
```bash
python -m src.preprocess_data
```

2. Start training:
```bash
python -m src.train
```

### Training on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/sauti_ya_kenya_training.ipynb)

Click the badge above to open our optimized training notebook in Google Colab. The notebook includes:

1. 🔍 Environment Setup
   - GPU verification
   - Drive mounting
   - Repository setup
   - Memory optimization

2. 📊 Data Management
   - Automatic data upload/extraction
   - Preprocessing verification
   - Directory structure setup

3. 🚀 Training with Memory Optimization
   - Batch size: 8 (memory-efficient)
   - Gradient accumulation: 4 steps
   - Automatic checkpoint management
   - GPU memory monitoring

4. 💾 Persistence
   - Checkpoints saved to Drive
   - Progress tracking
   - Easy resume capability

The notebook is designed to run efficiently on Colab's GPU while preventing out-of-memory errors.

### Resuming Training

To resume from a checkpoint:
```bash
python -m src.train --resume path/to/checkpoint.pt
```

## Using the API 🔌

### Quick Start

```python
from src.api import TTSEngine

# Initialize the TTS engine
engine = TTSEngine(model_path='checkpoints/best.pt')

# Generate speech
text = "Habari yako! How are you today?"
audio = engine.synthesize(text)

# Save to file
engine.save_audio(audio, 'output.wav')
```

### Advanced Usage

```python
# Custom configuration
engine = TTSEngine(
    model_path='checkpoints/best.pt',
    device='cuda',  # Use GPU if available
    sampling_rate=22050,
    language_mode='auto'  # Auto-detect language for code-switching
)

# Batch processing
texts = [
    "Niko na meeting leo jioni.",
    "Tutakutana kesho at the mall.",
    "Bei ni fifty shillings tu."
]

audios = engine.synthesize_batch(texts)

# Save multiple files
for i, audio in enumerate(audios):
    engine.save_audio(audio, f'output_{i}.wav')

# Stream audio directly
import sounddevice as sd

audio = engine.synthesize("Karibu tena!")
sd.play(audio, engine.sampling_rate)
sd.wait()  # Wait until audio finishes playing
```

### API Configuration

The TTS engine supports various configuration options:

```python
engine = TTSEngine(
    model_path='path/to/model',
    device='cuda',           # 'cuda' or 'cpu'
    sampling_rate=22050,     # Output audio sampling rate
    language_mode='auto',    # 'auto', 'swahili', 'mixed'
    speed_factor=1.0,       # Adjust speaking speed
    energy_factor=1.0,      # Adjust volume/energy
    pitch_factor=1.0,       # Adjust pitch
    use_cache=True          # Cache generated mel spectrograms
)
```

### Error Handling

```python
try:
    audio = engine.synthesize("Your text here")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Generation failed: {e}")
```

For more examples and detailed API documentation, see our [API Reference](docs/api.md).

## Project Structure 📁

```
Sauti-Ya-Kenya/
├── data/                     # Raw audio recordings and metadata
├── processed_data/           # Preprocessed training data
├── notebooks/               # Jupyter notebooks
│   └── sauti_ya_kenya_training.ipynb  # Optimized Colab training notebook
├── src/
│   ├── data_collection.py   # GUI tool for recording
│   ├── preprocess_data.py   # Data preprocessing pipeline
│   ├── model.py            # FastSpeech 2 model implementation
│   ├── train.py            # Training script with memory optimization
│   └── api.py              # Inference API
├── checkpoints/            # Model checkpoints
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community code of conduct
└── requirements.txt        # Project dependencies
```

## Model Architecture 🏗️

The TTS system uses FastSpeech 2 architecture with:
- Multi-head self-attention for text encoding
- Duration predictor for length regulation
- Mel-spectrogram decoder
- Parallel generation for fast inference

### Language-Specific Features

The model includes special handling for:
- Code-switching detection and processing
- Local expression preservation
- Proper nouns and place names
- Kenyan English accent patterns
- Common abbreviations and numerals

## 📞 Contact & Support

- **Email**: [information.msingiai@gmail.com](mailto:information.msingiai@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/Msingi-AI/Sauti-Ya-Kenya/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Msingi-AI/Sauti-Ya-Kenya/discussions)

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Setting up your development environment
- Recording voice data
- Submitting pull requests
- Code style guidelines

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Msingi AI](https://github.com/Msingi-AI)

</div>
