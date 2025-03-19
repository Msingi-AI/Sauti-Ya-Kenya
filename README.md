# Sauti Ya Kenya 🎙️

A Text-to-Speech (TTS) model for Kenyan Swahili using FastSpeech 2 architecture. This project aims to create a high-quality, natural-sounding voice synthesis system specifically designed for Kenyan Swahili speakers, capturing our unique expressions, code-switching patterns, and authentic accent.

## Features 🌟

- FastSpeech 2 architecture for fast, high-quality speech synthesis
- Support for Kenyan Swahili language features:
  - Natural code-switching between Swahili and English
  - Local expressions and idioms
  - Authentic Kenyan accent and intonation patterns
- User-friendly data collection tool with GUI interface
- Robust preprocessing pipeline
- GPU-accelerated training with checkpoint support
- Easy-to-use inference API

## Getting Started 🚀

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

1. Clone the repository:
```bash
git clone https://github.com/Msingi-AI/Sauti-Ya-Kenya.git
cd Sauti-Ya-Kenya
```

2. Install dependencies:
```bash
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

1. Open `colab_train.ipynb` in Google Colab
2. Mount your Google Drive
3. Upload your data.zip file when prompted
4. Training will automatically use GPU and save checkpoints

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
├── data/                  # Raw audio recordings and metadata
├── processed_data/        # Preprocessed training data
├── src/
│   ├── data_collection.py # GUI tool for recording
│   ├── preprocess_data.py # Data preprocessing pipeline
│   ├── model.py          # FastSpeech 2 model implementation
│   ├── train.py          # Training script
│   └── api.py            # Inference API
├── checkpoints/          # Model checkpoints
└── requirements.txt      # Project dependencies
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

## License 📝

[Insert your chosen license]

## Acknowledgments 🙏

- FastSpeech 2 paper authors
- Contributing voice talent
- Kenyan Swahili language experts
- [Add other acknowledgments]

## Contact 📧

[Add contact information]

---

Made with ❤️ by Msingi AI
