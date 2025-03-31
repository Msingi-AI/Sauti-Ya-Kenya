# Colab Training Notebooks

## Open in Colab
To open the training notebook in Colab:

1. Go to [Google Colab](https://colab.research.google.com)
2. Click File -> Upload Notebook
3. Select `train_tts_colab.ipynb` from your computer

Or use this direct link after pushing to GitHub:
```
https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/train_tts_colab.ipynb
```

## Setup Instructions
1. Mount your Google Drive
2. Create a folder structure in Drive:
   ```
   MyDrive/
   └── Sauti-Ya-Kenya/
       ├── data/
       │   ├── processed/  # Your processed audio data
       │   └── tokenizer/  # Trained tokenizer files
       └── checkpoints/    # Training checkpoints will be saved here
   ```
3. Copy your processed data and tokenizer files to the corresponding folders
4. Run all cells in sequence

## Features
- Memory-optimized training
- Automatic checkpoint saving
- Mixed precision training
- Gradient accumulation
- GPU monitoring
