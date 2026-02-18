import os
import modal

# 1. THE APP: 'App' is now the only way. 'Stub' will throw an error.
app = modal.App("sauti-distill")

# 2. THE VOLUME: Persistent storage for your activations and checkpoints.
# Use version=2 if you want the high-performance v2 volumes (recommended for 2026).
sauti_volume = modal.Volume.from_name("sauti-volume", create_if_missing=True)

# 3. THE IMAGE: All local code is now added EXPLICITLY to the Image.
# 'add_local_python_source' automounts your local 'src' package correctly.
image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
        "accelerate", "huggingface-hub", "numpy", "scipy", "wandb", "pyyaml"
    )
    .add_local_python_source("src") # <--- Canonical way to add your 'src' package
    .add_local_file("run_distill.py", remote_path="/root/run_distill.py") # Add the engine
    .add_local_dir("configs", remote_path="/root/configs") # Add your yaml configs
)

# 4. PATHS
VOLUME_PATH = "/root/data"

@app.function(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("hf-token"), 
        modal.Secret.from_name("wandb")
    ],
    volumes={VOLUME_PATH: sauti_volume},
    timeout=3600
)
def precompute_max_items(max_items: int = 2000):
    """
    Runs teacher precomputation.
    """
    # Environment variable mapping for Hugging Face
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Import is now safe because 'src' was added to the Image
    from sauti.precompute import precompute_teacher_activations

    out_dir = f"{VOLUME_PATH}/teacher_activations"
    print(f"🚀 Writing activations to persistent volume: {out_dir}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)
    
    # In Modal 1.0+, volumes must be committed to persist changes!
    sauti_volume.commit()
    print("✅ Volume committed. Data is safe.")


@app.function(
    image=image,
    gpu="A100",
    secrets=[
        modal.Secret.from_name("hf-token"), 
        modal.Secret.from_name("wandb")
    ],
    volumes={VOLUME_PATH: sauti_volume},
    timeout=86400 # 24 hour limit
)
def run_full_distill():
    """
    Runs the actual distillation engine.
    """
    os.environ["SAUTI_DATA_DIR"] = f"{VOLUME_PATH}/teacher_activations"
    
    print(f"🚀 Starting Distillation using data from: {os.environ['SAUTI_DATA_DIR']}")
    
    # Import the engine logic
    import run_distill
    run_distill.run_distillation()

if __name__ == "__main__":
    print("Usage: modal run modal_run.py ::precompute_max_items --max-items 200")