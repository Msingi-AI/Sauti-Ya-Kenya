import os
import sys
# ✅ FIX 1: Explicit imports to prevent 'AttributeError'
from modal import App, Image, Volume, Secret, Mount

# 1. DEFINITION
app = App("sauti-distill")

# 2. PERSISTENCE
# We want this to fail loudly if it can't find the volume
sauti_volume = Volume.from_name("sauti-volume", create_if_missing=True)

# 3. ENVIRONMENT
image = (
    Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
        "accelerate", "huggingface-hub", "numpy", "scipy", "wandb"
    )
)

# 4. PATHS
VOLUME_PATH = "/root/data"
PROJECT_PATH = "/root/project" 

@app.function(
    image=image,
    gpu="A100",
    secrets=[Secret.from_name("hf-token"), Secret.from_name("wandb")],
    # ✅ FIX 2: Mount local folder to /root/project
    mounts=[Mount.from_local_dir(".", remote_path=PROJECT_PATH)],
    volumes={VOLUME_PATH: sauti_volume},
    timeout=3600
)
def precompute_max_items(max_items: int = 2000):
    """
    Precompute teacher activations and save to the persistent Volume.
    """
    # ✅ FIX 3: Add project path to system so Python finds 'src'
    import sys
    sys.path.append(PROJECT_PATH) 

    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    from src.sauti.precompute import precompute_teacher_activations

    out_dir = f"{VOLUME_PATH}/teacher_activations"
    print(f"🚀 Writing activations to persistent volume: {out_dir}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)
    
    # Commit is automatic in newer Modal versions, but calling it doesn't hurt
    if hasattr(sauti_volume, "commit"):
        sauti_volume.commit()
    print("✅ Volume committed. Data is safe.")


@app.function(
    image=image,
    gpu="A100",
    secrets=[Secret.from_name("hf-token"), Secret.from_name("wandb")],
    mounts=[Mount.from_local_dir(".", remote_path=PROJECT_PATH)],
    volumes={VOLUME_PATH: sauti_volume},
    timeout=86400
)
def run_full_distill():
    """
    Run the full distillation using data from the Volume.
    """
    # ✅ FIX 3: Add project path here too
    import sys
    sys.path.append(PROJECT_PATH)

    os.environ["SAUTI_DATA_DIR"] = f"{VOLUME_PATH}/teacher_activations"
    
    print(f"🚀 Starting Distillation. Reading data from: {os.environ['SAUTI_DATA_DIR']}")
    
    from run_distill import run_distillation
    run_distillation()

if __name__ == "__main__":
    print("Usage: modal run modal_run.py ::precompute_max_items --max-items 2000")