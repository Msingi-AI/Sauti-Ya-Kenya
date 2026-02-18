import logging
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import get_waxal_swahili
from .distill_pipeline import append_manifest_record, deterministic_sample_id
from .security import is_remote_code_trusted

logger = logging.getLogger(__name__)


def precompute_teacher_activations(dataset_name, config_name, out_dir, max_items=2000, manifest_filename="manifest.jsonl"):
    """Run the teacher model on dataset samples and persist hidden states + manifest."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_id = "fishaudio/fish-speech-1.5"
    logger.info("Using device=%s teacher=%s", device, teacher_id)

    tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=is_remote_code_trusted(teacher_id))
    teacher = AutoModel.from_pretrained(teacher_id, trust_remote_code=is_remote_code_trusted(teacher_id)).to(device)
    teacher.eval()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / manifest_filename

    ds = get_waxal_swahili(split="train", streaming=True, dataset_name=dataset_name, config_name=config_name)
    logger.info("Processing up to %s items", max_items)

    count = 0
    with torch.no_grad():
        for sample in tqdm(ds):
            if count >= max_items:
                break

            text = sample.get("text", "")
            audio = sample.get("audio") or {}
            audio_path = audio.get("path", "") if isinstance(audio, dict) else ""
            sample_id = deterministic_sample_id(text=text, audio_path=audio_path)
            save_path = out_path / f"{sample_id}.pt"

            if save_path.exists():
                continue

            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = teacher(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state.cpu()

            torch.save(
                {
                    "sample_id": sample_id,
                    "hidden_states": hidden_states,
                    "text": text,
                    "audio_path": audio_path,
                },
                save_path,
            )
            append_manifest_record(
                str(manifest_path),
                sample_id,
                {"text": text, "audio_path": audio_path, "tensor_file": save_path.name},
            )
            count += 1

    logger.info("Precomputation complete. Saved %s files to %s", count, out_dir)
