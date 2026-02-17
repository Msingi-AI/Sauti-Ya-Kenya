Modal job scaffolding instructions
===============================

This project includes a small Modal runner `modal_run.py` that exposes two functions:

- `precompute_max_items(max_items)` — precomputes teacher activations and writes them to `checkpoints/teacher_activations`.
- `run_full_distill()` — runs the distillation script `run_distill.py`, reading cached activations and saving checkpoints to `checkpoints/`.

Prerequisites
-------------

- Install the Modal CLI / Python SDK: https://modal.com/docs
- Have a Modal account and credits (you mentioned you have credits).
- Store your Hugging Face token in Modal secrets as `hf-token` (see below).

Create the secret
-----------------

Use the Modal dashboard or CLI to add your Hugging Face token as a secret named `hf-token`.

Example (Modal CLI):

1. In the Modal dashboard: create a new Secret with name `hf-token` and paste your HF token.
2. Or with CLI (if available): `modal secret create hf-token` and follow prompts.

Run precompute (teacher activations)
-----------------------------------

This is the recommended first step: run the heavy teacher forward on a large GPU and persist activations.

Local command (runs remotely on Modal):

```bash
modal run modal_run.py precompute_max_items --max-items 2000
```

Adjust `--max-items` to control how many examples you precompute in a single job.

Run full distillation
---------------------

After activations are stored (in the mounted `checkpoints/teacher_activations`), run the distillation job:

```bash
modal run modal_run.py run_full_distill
```

Notes & recommendations
-----------------------

- GPU selection: prefer `H100` or `A100-80GB` for the teacher precompute step. The runner uses `A100-40GB` by default; change the decorator `gpu(...)` in `modal_run.py` if you need a different GPU.
- Image: the scaffold uses pip-installed packages. For production, create a custom CUDA-enabled image with a matching PyTorch wheel for the target GPU.
 - Image: the scaffold uses pip-installed packages. For production, create a custom CUDA-enabled image with a matching PyTorch wheel for the target GPU. A sample `Dockerfile` is included at the repo root; adjust the `torch` wheel line to pin the wheel matching your target CUDA (Modal often documents the recommended wheel per GPU family).
- Mounts: the mounting strategy in `modal_run.py` uses the current repo directory. This means outputs (checkpoints, activations) will be written into `./checkpoints` on your machine when the Modal job mounts that directory.
- Secrets: never hardcode tokens; use Modal secrets only.

If you want, I can add a Dockerfile with pinned CUDA/PyTorch wheels and a more production-ready Modal Image specification next.
