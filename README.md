# Sauti — by MsingiAI

Sauti is the new home for MsingiAI's Swahili-first speech efforts. This repository contains scaffolding for dataset preparation, knowledge distillation, and fine-tuning pipelines targeted at high-quality, efficient Swahili TTS models (WAXAL-compatible).

Quickstart

- Install dependencies:

```bash
pip install -r requirements-sauti.txt
```

- Prepare the WAXAL dataset (local path or Hugging Face): see `src/sauti/data.py`.

- Run the distillation scaffold example:

```bash
python examples/run_distill.py --help
```

See the `src/sauti` package for core modules: data handling, distillation, finetuning, and inference.

License: Check `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for community and usage guidelines.
