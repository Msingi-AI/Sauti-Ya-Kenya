import os
import modal

# 1. THE APP
app = modal.App("sauti-distill")

# 2. THE VOLUME
sauti_volume = modal.Volume.from_name("sauti-volume", create_if_missing=True)

# 3. THE IMAGE (Updated for proper imports)
image = (
    modal.Image.debian_slim()
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
        "accelerate", "huggingface-hub", "numpy", "scipy", "wandb", "pyyaml"
    )
    # This is the magic line. It finds the local 'src' folder 
    # and makes 'sauti' importable as 'from sauti.xxx'
    .add_local_python_source("src") 
    .add_local_file("run_distill.py", remote_path="/root/run_distill.py")
    .add_local_dir("configs", remote_path="/root/configs")
)

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
    if "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Try importing the `sauti` package. If it isn't found, add common
    # mounted paths to sys.path so the package can be imported from the repo.
    try:
        from sauti.precompute import precompute_teacher_activations
    except Exception as e:
        import sys
        print("Initial import failed:", e)
        # Debug sys.path and available directories
        try:
            print("sys.path before fix:", sys.path)
        except Exception:
            pass

        # Common mount locations Modal may use: /root/project/src, /root/src, /root/project
        candidate_paths = ["/root/project/src", "/root/src", "/root/project", "/root"]
        for p in candidate_paths:
            if os.path.exists(p) and p not in sys.path:
                print(f"Adding {p} to sys.path")
                sys.path.insert(0, p)

        # Also try adding the sauti package dir directly if present
        sauti_dir = "/root/src/sauti"
        if os.path.exists(sauti_dir) and sauti_dir not in sys.path:
            print(f"Adding {sauti_dir} to sys.path")
            sys.path.insert(0, sauti_dir)

        # Retry import
        try:
            from sauti.precompute import precompute_teacher_activations
        except Exception as e2:
            print("Retry import failed:", e2)
            # Show directory listings for debugging
            try:
                print("/root listing:", os.listdir("/root"))
                if os.path.exists("/root/project"):
                    print("/root/project listing:", os.listdir("/root/project"))
                if os.path.exists("/root/project/src"):
                    print("/root/project/src listing:", os.listdir("/root/project/src"))
            except Exception as de:
                print("Listing error:", de)
            raise

    out_dir = f"{VOLUME_PATH}/teacher_activations"
    print(f"🚀 Writing activations to persistent volume: {out_dir}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)
    
    sauti_volume.commit()
    print("✅ Volume committed. Data is safe.")