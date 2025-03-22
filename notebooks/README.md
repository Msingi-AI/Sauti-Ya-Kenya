# Sauti ya Kenya Notebooks

## Tokenizer Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/train_tokenizer_v2.ipynb)

This notebook trains the Swahili tokenizer with code-switching support using Colab's free GPU.

### Instructions:
1. Click the "Open in Colab" button above
2. Upload your `archive.zip` when prompted in the notebook
3. The trained tokenizer will be automatically saved to:
   ```
   /content/drive/MyDrive/Sauti-Ya-Kenya/tokenizer/tokenizer.model
   ```

### Features:
- Uses Colab's T4 GPU for faster training
- Simple file upload widget - no manual Drive setup needed
- Handles code-switched text (Swahili-English)
- Shows token count and sample files before training
- Tests the tokenizer on sample sentences
- Saves the model to Google Drive
