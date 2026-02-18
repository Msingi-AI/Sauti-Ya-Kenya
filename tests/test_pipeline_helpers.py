import importlib.util

import pytest

from src.sauti.distill_pipeline import deterministic_sample_id
from src.sauti.security import is_remote_code_trusted


def test_deterministic_sample_id_is_stable():
    sid_1 = deterministic_sample_id("habari", "path.wav")
    sid_2 = deterministic_sample_id("habari", "path.wav")
    sid_3 = deterministic_sample_id("habari", "other.wav")

    assert sid_1 == sid_2
    assert sid_1 != sid_3


@pytest.mark.skipif(importlib.util.find_spec("yaml") is None, reason="PyYAML not installed")
def test_runtime_config_respects_env_overrides(tmp_path):
    from src.sauti.distill_pipeline import DistillRuntimeConfig

    cfg = tmp_path / "distill.yaml"
    cfg.write_text(
        """
project:
  output_dir: ./from-config
data:
  train_dataset: google/WaxalNLP
  subset: swa_tts
teacher:
  model_id: fishaudio/fish-speech-1.5
  hidden_size: 1024
student:
  model_id: FunAudioLLM/CosyVoice2-0.5B
  hidden_size: 512
training:
  learning_rate: 0.0001
  max_epochs: 5
distillation:
  alpha_hard: 0.5
  alpha_mel: 1.0
  temperature: 2.0
""",
        encoding="utf-8",
    )

    runtime = DistillRuntimeConfig.from_yaml(
        str(cfg),
        env={"SAUTI_OUTPUT_DIR": "./from-env", "SAUTI_DATA_DIR": "./precomputed", "SAUTI_MAX_STEPS": "12"},
    )

    assert runtime.output_dir == "./from-env"
    assert runtime.data_dir == "./precomputed"
    assert runtime.max_steps == 12


def test_remote_code_policy_precedence():
    assert is_remote_code_trusted("any/id", env={"SAUTI_TRUST_REMOTE_CODE": "true"})
    assert not is_remote_code_trusted("fishaudio/fish-speech-1.5", env={"SAUTI_TRUST_REMOTE_CODE": "false"})
    assert is_remote_code_trusted("custom/id", env={"SAUTI_REMOTE_CODE_ALLOWLIST": "custom/id"})
