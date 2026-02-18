import modal
import os

# 1. DEFINITION: Use 'App' instead of 'Stub'
app = modal.App("sauti-distill")

# 2. PERSISTENCE: Create a Volume to store checkpoints and data
# This is like a shared hard drive that persists between your functions.
sauti_volume = modal.Volume.from_name("sauti-volume", create_if_missing=True)

# 3. ENVIRONMENT: Add 'libsndfile1' and 'ffmpeg' for audio processing
image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")  # <--- CRITICAL FIX
    .pip_install(
        "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
        "accelerate", "huggingface-hub", "numpy", "scipy", "wandb"
    )
)

# 4. PATHS: Define where the volume lives inside the container
VOLUME_PATH = "/root/data"

@app.function(
    image=image,
    gpu="A100", # Modal will auto-select memory variant, or specify "A100-40GB"
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")],
    # Mount local code for imports, but stick data in the Volume
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/project")],
    volumes={VOLUME_PATH: sauti_volume}, # <--- Mounts the persistent drive
    timeout=3600 # 1 hour timeout for safety
)
def precompute_max_items(max_items: int = 2000):
    """
    Precompute teacher activations and save to the persistent Volume.
    """
    # Verify secrets are loaded (Modal sets env vars automatically if secret is created correctly)
    # If you named the secret key 'HF_TOKEN', you might need to map it:
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    from src.sauti.precompute import precompute_teacher_activations

    # Write to the VOLUME, not the ephemeral project folder
    out_dir = f"{VOLUME_PATH}/teacher_activations"
    print(f"🚀 Writing activations to persistent volume: {out_dir}")
    
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)
    
    # Commit the volume to ensure data is saved immediately
    sauti_volume.commit()
    print("✅ Volume committed. Data is safe.")


@app.function(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")],
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/project")],
    volumes={VOLUME_PATH: sauti_volume}, # <--- Mount the SAME volume here
    timeout=86400 # 24 hours for full training
)
def run_full_distill():
    """
    Run the full distillation using data from the Volume.
    """
    import sys
    
    # Point your training script to look for data in the Volume
    # You might need to update your 'run_distill.py' to accept a data_dir argument
    # or set an environment variable.
    os.environ["SAUTI_DATA_DIR"] = f"{VOLUME_PATH}/teacher_activations"
    
    print(f"🚀 Starting Distillation. Reading data from: {os.environ['SAUTI_DATA_DIR']}")
    
    # Run the distillation
    from run_distill import run_distillation
    run_distillation()

if __name__ == "__main__":
    print("Usage: modal run modal_run.py ::precompute_max_items --max-items 2000")