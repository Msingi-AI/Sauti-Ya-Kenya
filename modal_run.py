import modal
import os

# Build compatibility fallbacks for different modal SDK versions
app = getattr(modal, "App")("sauti-distill") if hasattr(modal, "App") else getattr(modal, "stub", None)

# Try to create or find a Volume; if Volume API is missing, continue without it
sauti_volume = None
VOLUME_PATH = "/root/data"
if hasattr(modal, "Volume"):
    try:
        sauti_volume = modal.Volume.from_name("sauti-volume", create_if_missing=True)
    except Exception:
        sauti_volume = None

# Build image (likely present across versions)
image = modal.Image.debian_slim().apt_install("libsndfile1", "ffmpeg").pip_install(
    "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
    "accelerate", "huggingface-hub", "numpy", "scipy", "wandb"
)

# Prepare decorator kwargs with graceful fallbacks
decorator_kwargs = {
    "image": image,
    "gpu": "A100",
}

# Secrets
if hasattr(modal, "Secret") and hasattr(modal.Secret, "from_name"):
    try:
        decorator_kwargs["secrets"] = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]
    except Exception:
        pass

# Mounts
mounts_obj = None
if hasattr(modal, "Mount") and hasattr(modal.Mount, "from_local_dir"):
    try:
        mounts_obj = [modal.Mount.from_local_dir(".", remote_path="/root/project")]
        decorator_kwargs["mounts"] = mounts_obj
    except Exception:
        mounts_obj = None

# Volumes
if sauti_volume is not None:
    try:
        decorator_kwargs["volumes"] = {VOLUME_PATH: sauti_volume}
    except Exception:
        pass

# Timeouts
decorator_kwargs.setdefault("timeout", 3600)


@app.function(**decorator_kwargs)
def precompute_max_items(max_items: int = 2000):
    """
    Precompute teacher activations and save to the persistent Volume.
    """
    # Verify secrets are loaded (Modal sets env vars automatically if secret is created correctly)
    # If you named the secret key 'HF_TOKEN', you might need to map it:
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    import sys
    # Ensure the mounted project directory is on sys.path in the Modal container
    project_root = "/root/project"
    if project_root not in sys.path and os.path.exists(project_root):
        sys.path.insert(0, project_root)

    from src.sauti.precompute import precompute_teacher_activations

    # Write to the VOLUME, not the ephemeral project folder
    out_dir = f"{VOLUME_PATH}/teacher_activations"
    print(f"🚀 Writing activations to persistent volume: {out_dir}")
    
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)

    # Commit the volume if available
    if sauti_volume is not None and hasattr(sauti_volume, "commit"):
        try:
            sauti_volume.commit()
            print("✅ Volume committed. Data is safe.")
        except Exception:
            print("⚠️ Volume commit failed or not supported in this SDK version.")


decorator_kwargs_full = {
    "image": image,
    "gpu": "A100",
    "timeout": 86400,
}

if hasattr(modal, "Secret") and hasattr(modal.Secret, "from_name"):
    try:
        decorator_kwargs_full["secrets"] = [modal.Secret.from_name("hf-token"), modal.Secret.from_name("wandb")]
    except Exception:
        pass

if mounts_obj is not None:
    decorator_kwargs_full["mounts"] = mounts_obj

if sauti_volume is not None:
    try:
        decorator_kwargs_full["volumes"] = {VOLUME_PATH: sauti_volume}
    except Exception:
        pass


@app.function(**decorator_kwargs_full)
def run_full_distill():
    """
    Run the full distillation using data from the Volume.
    """
    import sys

    # Ensure the mounted project directory is on sys.path in the Modal container
    project_root = "/root/project"
    if project_root not in sys.path and os.path.exists(project_root):
        sys.path.insert(0, project_root)

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