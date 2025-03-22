# Sauti ya Kenya Notebooks

## Tokenizer Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/train_tokenizer_from_archive.ipynb)

This notebook trains the Swahili tokenizer with code-switching support using Colab's free GPU.

### Prerequisites:
1. Upload your `archive.zip` containing training text to:
   ```
   /content/drive/MyDrive/Sauti-Ya-Kenya/data/archive.zip
   ```

2. The trained tokenizer will be saved to:
   ```
   /content/drive/MyDrive/Sauti-Ya-Kenya/tokenizer/tokenizer.model
   ```

### Features:
- Uses Colab's T4 GPU for faster training
- Handles code-switched text (Swahili-English)
- Automatically extracts text from archive.zip
- Tests the tokenizer on sample sentences
- Saves the model to Google Drive
