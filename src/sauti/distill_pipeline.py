import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from .security import is_remote_code_trusted

logger = logging.getLogger(__name__)


@dataclass
class DistillRuntimeConfig:
    teacher_model_id: str
    student_model_id: str
    output_dir: str
    learning_rate: float
    max_epochs: int
    hidden_teacher: int
    hidden_student: int
    alpha_feat: float = 1.0
    alpha_hard: float = 0.5
    temperature: float = 2.0
    max_steps: int = 100
    data_dir: Optional[str] = None
    dataset_name: str = "google/WaxalNLP"
    dataset_config: str = "swa_tts"

    @classmethod
    def from_yaml(cls, path: str, env: Optional[dict] = None) -> "DistillRuntimeConfig":
        env_map = env or os.environ
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        output_dir = env_map.get("SAUTI_OUTPUT_DIR", cfg["project"]["output_dir"])
        data_dir = env_map.get("SAUTI_DATA_DIR")
        max_steps = int(env_map.get("SAUTI_MAX_STEPS", "100"))

        return cls(
            teacher_model_id=cfg["teacher"]["model_id"],
            student_model_id=cfg["student"]["model_id"],
            output_dir=output_dir,
            learning_rate=float(cfg["training"]["learning_rate"]),
            max_epochs=int(cfg["training"]["max_epochs"]),
            hidden_teacher=int(cfg["teacher"]["hidden_size"]),
            hidden_student=int(cfg["student"]["hidden_size"]),
            temperature=float(cfg.get("distillation", {}).get("temperature", 2.0)),
            alpha_hard=float(cfg.get("distillation", {}).get("alpha_hard", 0.5)),
            alpha_feat=float(cfg.get("distillation", {}).get("alpha_mel", 1.0)),
            max_steps=max_steps,
            data_dir=data_dir,
            dataset_name=cfg["data"]["train_dataset"],
            dataset_config=cfg["data"]["subset"],
        )


class TeacherStateProvider:
    def __init__(self, data_dir: Optional[str], device):
        self.data_dir = data_dir
        self.device = device

    def get_state(self, sample_id: str) -> Optional[object]:
        if not self.data_dir:
            return None
        fpath = Path(self.data_dir) / f"{sample_id}.pt"
        if not fpath.exists():
            return None
        import torch

        data = torch.load(fpath, map_location=self.device)
        return data["hidden_states"]


def deterministic_sample_id(text: str, audio_path: str = "") -> str:
    key = f"{text.strip()}::{audio_path.strip()}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"sample_{digest}"


def iter_samples(dataset: Iterable[Dict]) -> Iterator[Tuple[str, Dict]]:
    for sample in dataset:
        text = sample.get("text", "")
        audio = sample.get("audio") or {}
        audio_path = audio.get("path", "") if isinstance(audio, dict) else ""
        yield deterministic_sample_id(text=text, audio_path=audio_path), sample


def run_distillation(config_path: str = "configs/distill.yaml"):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    from .data import get_waxal_swahili
    from .distillation_loss import SautiDistillationLoss

    cfg = DistillRuntimeConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Launching distillation on %s", device)
    logger.info("Remote code policy: student=%s teacher=%s", is_remote_code_trusted(cfg.student_model_id), is_remote_code_trusted(cfg.teacher_model_id))

    student = AutoModel.from_pretrained(
        cfg.student_model_id,
        trust_remote_code=is_remote_code_trusted(cfg.student_model_id),
    ).to(device)
    student.train()

    teacher = None
    provider = TeacherStateProvider(cfg.data_dir, device)
    if not cfg.data_dir:
        teacher = AutoModel.from_pretrained(
            cfg.teacher_model_id,
            trust_remote_code=is_remote_code_trusted(cfg.teacher_model_id),
        ).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    projection = nn.Linear(cfg.hidden_teacher, cfg.hidden_student).to(device)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projection.parameters()), lr=cfg.learning_rate)
    criterion = SautiDistillationLoss(temperature=cfg.temperature, alpha_feat=cfg.alpha_feat, alpha_hard=cfg.alpha_hard)

    ds = get_waxal_swahili(
        split="train",
        streaming=True,
        dataset_name=cfg.dataset_name,
        config_name=cfg.dataset_config,
    )

    step = 0
    for epoch in range(cfg.max_epochs):
        for sample_id, sample in iter_samples(ds):
            if step >= cfg.max_steps:
                break
            text = sample.get("text", "")
            if len(text) < 2:
                continue

            t_hidden = provider.get_state(sample_id)
            if t_hidden is None and teacher is not None:
                # Fallback intentionally explicit: if live teacher path is not wired yet, skip.
                continue
            if t_hidden is None:
                continue

            # Placeholder stage remains explicit until student tokenizer/forward contract is finalized.
            if step % 10 == 0:
                logger.info("Epoch %s | Step %s | sample_id=%s", epoch, step, sample_id)
            step += 1

        if step >= cfg.max_steps:
            logger.info("Smoke run complete")
            break

        save_path = Path(cfg.output_dir) / f"epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(str(save_path))
        torch.save(projection.state_dict(), save_path / "projection.pt")

    return {"steps": step, "output_dir": cfg.output_dir}


def append_manifest_record(manifest_path: str, sample_id: str, metadata: Dict):
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"sample_id": sample_id, **metadata}
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
