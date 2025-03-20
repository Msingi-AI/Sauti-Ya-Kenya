# 🎙️ Sauti Ya Kenya - Colab Training

Click the badge below to open in Colab (opens in new tab):

<a href="https://colab.research.google.com/github/Msingi-AI/Sauti-Ya-Kenya/blob/main/notebooks/train_tts_model.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This notebook provides GPU-accelerated training for the Sauti Ya Kenya TTS model with optimized memory usage. It will:

1. 🔄 Mount your Google Drive for persistent storage
2. 📥 Clone the latest code from GitHub
3. ⚙️ Set up the training environment with memory optimization
4. 📊 Load and preprocess your data
5. 🚀 Train the model with automatic checkpointing
6. 📈 Monitor GPU memory usage and training progress

## Memory Optimization Features

- Reduced batch size (8) for efficient memory usage
- Gradient accumulation (4 steps)
- Automatic CUDA cache cleanup
- Memory allocation monitoring
- Checkpoint management in Drive

## Getting Started

1. Upload `data.zip` to `/content/drive/MyDrive/Sauti-Ya-Kenya/`
2. Click the Colab badge above (opens in new tab)
3. Connect to GPU runtime (Runtime → Change runtime type → GPU)
4. Run cells in order
5. Monitor training progress in the output
