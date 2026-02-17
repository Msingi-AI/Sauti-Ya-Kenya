from modal import Stub, Image, gpu, Secret, Mount

stub = Stub("sauti-distill")

# Use a minimal image that installs required Python packages. For production
# use a CUDA-enabled wheel matching your target GPU (Modal provides CUDA images).
image = (
    Image.debian_slim()
    .pip_install(
        "torch", "transformers", "datasets", "torchaudio", "librosa", "soundfile",
        "accelerate", "huggingface-hub", "numpy", "scipy"
    )
)


@stub.function(
    image=image,
    gpu=gpu("A100-40GB"),
    secret=Secret.from_name("hf-token"),
    mounts=[Mount.from_local_dir(".", "/root/project")],
)
def precompute_max_items(max_items: int = 2000):
    """Precompute teacher activations on Modal and write into mounted `/root/project/checkpoints`.

    Usage (local):
      modal run modal_run.py precompute_max_items --max-items 2000
    """
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = Secret.from_name("hf-token").get()
    from src.sauti.precompute import precompute_teacher_activations

    out_dir = "/root/project/checkpoints/teacher_activations"
    precompute_teacher_activations("google/WaxalNLP", "swa_tts", out_dir=out_dir, max_items=max_items)


@stub.function(
    image=image,
    gpu=gpu("A100-40GB"),
    secret=Secret.from_name("hf-token"),
    mounts=[Mount.from_local_dir(".", "/root/project")],
)
def run_full_distill():
    """Run the full distillation script on Modal. Assumes `checkpoints/teacher_activations` exists.

    Usage (local):
      modal run modal_run.py run_full_distill
    """
    import os
    os.environ["HUGGINGFACE_HUB_TOKEN"] = Secret.from_name("hf-token").get()
    # Import and run the distillation runner from the repository
    from run_distill import run_distillation
    run_distillation()


if __name__ == "__main__":
    print("Modal runner: call functions via the Modal CLI, e.g. `modal run modal_run.py precompute_max_items` or `modal run modal_run.py run_full_distill`.")
